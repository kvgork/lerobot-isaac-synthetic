"""
mimicgen sub-package — DEFERRED PATH
======================================
Thin wrapper over the ``lerobot_mimicgen_bridge`` skill for MimicGen-based
synthetic data augmentation.

Status: **deferred** — MimicGen runs in MuJoCo/robosuite internally and is not
integrated with Isaac Lab.  This path produces LeRobot Parquet episodes that can
be merged with Isaac Lab DR episodes via ``merge_utilities``.

Activation
----------
Set the environment variable ``LEROBOT_MIMICGEN_ENABLED=1`` or pass
``enabled=True`` to ``bridge_invocation.run_mimicgen`` to activate.  By default
all functions raise ``NotImplementedError`` pointing to the skill and agent.

Skill reference (conversion, full implementation):
  ${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md

Agent reference (full pipeline orchestration):
  ${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md

Deferred-path note:
  The blocker for this path is that MimicGen v1.x only supports robosuite/MuJoCo
  task definitions.  Once a MimicGen → Isaac Lab bridge is available, this stub
  can be replaced with a real implementation.
"""
