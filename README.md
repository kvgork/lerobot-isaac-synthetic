# lerobot-isaac-synthetic

Synthetic data generation for the LeRobot + Isaac Lab training workspace.

Provides two paths to expand the training corpus beyond real SO-101 teleoperation data:

---

## Purpose

`lerobot-isaac-synthetic` generates synthetic `LeRobotDataset`-compatible Parquet data
that can be merged with real teleoperation recordings to improve policy robustness. The
package implements two data augmentation pipelines:

1. **Isaac Lab Domain Randomization (DR) replay** — the priority path. Replays real
   teleoperated episodes through an Isaac Lab environment whose DR parameters are
   re-sampled on each reset, producing variants tagged `source="sim_dr"`.
2. **MimicGen bridge** — the deferred path. Raises `NotImplementedError` by default;
   activated by setting `LEROBOT_MIMICGEN_ENABLED=1` and implementing the stub.

Both paths produce output with the same column schema as real SO-101 data, so policies
can be trained on merged datasets without any schema conversion.

---

## Status

| Component | Status |
|-----------|--------|
| `isaac_dr/replay_runner.py` | Implemented — requires Isaac Lab + lerobot at call time |
| `isaac_dr/parquet_writer.py` | Implemented — requires lerobot + pyarrow |
| `merge_utilities.py` | Implemented — requires pandas + pyarrow + lerobot |
| `mimicgen/bridge_invocation.py` | Deferred stub — `NotImplementedError`; see plan §4b |

---

## Synthetic Data Paths

### Path 1 — Isaac Lab Domain Randomization (priority)

Replay recorded teleoperation episodes through an Isaac Lab environment whose
domain-randomization (DR) parameters are re-sampled on each reset. Each real
episode produces N synthetic variants with randomized:

- Object pose (±5 cm, stage-dependent)
- Table surface friction
- Lighting intensity and direction
- Wrist/overhead camera field-of-view jitter
- Joint friction

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
    --source_dataset /data/real \
    --output_path /data/synthetic_dr \
    --n_variants 5 \
    --env_id Isaac-SO101-PickPlace-v0 \
    --dry_run
```

**Prerequisites:**
- Isaac Lab installed system-wide
- `lerobot-isaac-env` package (`Isaac-SO101-PickPlace-v0` gym registration)
- `lerobot` installed: `pip install lerobot`
- `pyarrow` and `pandas`: included in pyproject.toml dependencies

---

### Path 2 — MimicGen Bridge (deferred)

**Status: stub only — raises `NotImplementedError` by default.**  
**Deferred — see plan §4b.**

```python
from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen
# Raises NotImplementedError — use Isaac Lab DR path or set LEROBOT_MIMICGEN_ENABLED=1
```

To enable: set `LEROBOT_MIMICGEN_ENABLED=1` AND install MimicGen + robosuite,
then implement the stub body. Recommended: invoke via `lerobot-sim-augmentation-agent`
for full pipeline with error recovery.

Skill reference: `${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md`
Agent reference: `${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md`

---

## Merge Utilities

Combine real, DR, and MimicGen datasets into one unified `LeRobotDataset`:

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

merged_path = merge_datasets(
    real_path="/data/real",
    sim_paths=["/data/synthetic_dr", "/data/synthetic_mimicgen"],
    output_path="/data/merged",
    sim_weight=0.5,   # 50% real, 50% synthetic
)
```

The merged dataset:
- Re-assigns contiguous `episode_index` values starting from 0
- Preserves `source` column (`"real"` / `"sim_dr"` / `"mimicgen"`) in metadata
- Deduplicates near-identical episodes (compares first-frame `observation.state` hash)
- Writes `meta/stats.json` and `meta/info.json` via `LeRobotDataset.consolidate()`

---

## Dataset Schema Invariant

All Parquet output matches the SO-101 real-teleoperation column schema:

| Column | dtype | shape |
|--------|-------|-------|
| `observation.state` | float32 | (12,) — joint pos (6) + vel (6) |
| `observation.images.wrist` | binary (video) | 480×640×3 |
| `observation.images.overhead` | binary (video) | 480×640×3 |
| `action` | float32 | (6,) — joint position targets (radians) |
| `next.done` | bool | (1,) |

---

## Public API

- **`replay_with_randomization(...)`** — generator yielding `Episode` objects
- **`Episode`** — dataclass: `episode_index`, `source_episode_index`, `dr_seed`,
  `observations`, `actions`, `success`, `metadata`
- **`write_episodes_to_lerobot_dataset(episodes, output_path, ...)`** — writes
  `Episode` iterable to a LeRobotDataset-compatible Parquet directory
- **`merge_datasets(real_path, sim_paths, output_path, sim_weight=0.5, ...)`** — merges
  real + synthetic datasets with balanced sampling
- **`run_mimicgen(...)`** — deferred stub; `NotImplementedError` by default

---

## Installation

```bash
# From workspace root
pip install -e packages/lerobot-isaac-synthetic

# With dev extras
pip install -e "packages/lerobot-isaac-synthetic[dev]"
```

---

## Dependencies

### Python (pyproject.toml)

```
pyarrow>=14.0
pandas>=2.0
numpy>=1.24
```

### Heavy/external dependencies

| Dependency | Required for | Install |
|------------|-------------|---------|
| Isaac Lab | `replay_with_randomization()` | system-wide via Isaac Lab installer |
| `lerobot` | `replay_with_randomization()`, `parquet_writer`, `merge_datasets` | `pip install lerobot` |
| `lerobot-isaac-env` | `replay_with_randomization()` gym registration | `pip install -e packages/lerobot-isaac-env/` |
| MimicGen | `run_mimicgen()` (deferred) | see MimicGen docs |
| robosuite | `run_mimicgen()` (deferred) | see robosuite docs |

All heavy deps are **soft-imported** — the package imports cleanly without them.

---

## Configuration

DR parameters are passed as a `dr_config` dict to `replay_with_randomization()`:

| Key | Type | Effect |
|-----|------|--------|
| `object_pose_noise_m` | `float` | Positional noise in metres |
| `lighting_variant` | `bool` | Enable lighting randomisation |
| `table_friction_range` | `tuple` | (min, max) friction coefficients |
| `camera_fov_jitter_deg` | `float` | FOV jitter in degrees |

---

## Running Tests

```bash
pytest packages/lerobot-isaac-synthetic/tests/ -v
```

All tests run without Isaac Lab, lerobot, or MimicGen (soft imports).

---

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-synthetic -b spinout-synthetic
git checkout spinout-synthetic
git remote add origin git@github.com:user/lerobot-isaac-synthetic.git
git push -u origin main
```

After spinout: `lerobot-isaac-env` and `lerobot-isaac-configs` become external pip deps.
See `../../docs/ARCHITECTURE.md` — spinout section.
