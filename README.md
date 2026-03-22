# Hierarchical VLM+RL Pick-and-Place for G1 Humanoid

End-to-end autonomous pick-and-place on the Unitree G1 (29 DoF) in NVIDIA Isaac Lab.
A natural-language command drives a VLM planner that sequences RL skill primitives
-- walk, reach, grasp, lift, carry, lower, place -- executed by cascaded locomotion
and manipulation policies trained entirely in simulation.

https://github.com/user-attachments/assets/placeholder-video-link

## Key Results

- **8/8 skill steps** completed autonomously in ~90 s (sim time)
- **Zero falls** during full pick-carry-place trajectory
- Lateral carry walk with Pure Pursuit controller (no rotation, pure strafe)
- Magnetic grasp attaches at 0.21 m with orientation preservation
- Trained with 12-level curriculum (perturbation-robust locomotion)

## Architecture

```
                        ┌────────────────────────────────────────────────┐
  "Pick up the          │              Skill Executor                   │
   steering wheel" ───► │  pre_reach ► walk_to ► reach ► grasp ► lift  │
                        │  ► carry_walk ► lower ► place                │
                        └─────────┬──────────────┬──────────────────────┘
                                  │              │
                          ┌───────▼──────┐ ┌─────▼──────┐
                          │  Loco Policy │ │ Arm Policy  │
                          │  (Stage 2)   │ │ (Stage 2)   │
                          │  66 → 15     │ │ 39 → 7      │
                          │  50 Hz       │ │ 50 Hz       │
                          └───────┬──────┘ └─────┬──────┘
                                  │              │
                          ┌───────▼──────────────▼──────┐
                          │   Isaac Lab / PhysX 50 Hz   │
                          │   G1 29-DoF  (12L+3W+7A+7H) │
                          └─────────────────────────────┘
```

### Triple-Policy Cascade

| Policy | Input | Output | Training |
|--------|-------|--------|----------|
| **Stage 2 Loco** | 66-dim (body vel, gravity, joints, commands) | 15-dim (12 leg + 3 waist) | V6.2 fine-tuned with arm perturbation, 12-level curriculum |
| **Stage 2 Arm** | 39-dim (arm joints, EE body-frame, target, orient) | 7-dim (right arm joints) | Frozen loco, spherical workspace reaching |
| **Finger** | Heuristic | 14-dim (7 per hand) | DEX3 open/close controller |

Each policy is trained with the previous frozen, enabling sequential curriculum learning.

## Demo

```bash
cd C:\IsaacLab
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py ^
    --num_envs 1 ^
    --checkpoint <stage2_loco_checkpoint> ^
    --arm_checkpoint <stage2_arm_checkpoint> ^
    --task "Pick up the steering wheel from the table" ^
    --planner simple
```

Add `--record` to save a video of the run.

### Skill Execution Pipeline

```
Step 1  pre_reach    Raise arm to high position (EE z ≈ 0.97 m)
Step 2  walk_to      Walk to object (Pure Pursuit, ~300 steps)
Step 3  reach        Extend arm to object (Stage 2 arm policy)
Step 4  grasp        Magnetic grasp attachment (0.21 m threshold)
Step 5  lift         Lift object (EE z ≈ 0.95 m)
Step 6  walk_to      Lateral carry walk to basket (Pure Pursuit, vyaw = 0)
Step 7  lower        Lower arm to table height
Step 8  place        Release object into basket
```

## Training

### Stage 2 Locomotion (Perturbation-Robust)

```bash
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\29dof\train_unified_stage_2_loco.py ^
    --stage1_checkpoint <v6.2_model.pt> ^
    --arm_checkpoint <stage2_arm_model.pt> ^
    --num_envs 4096 --headless
```

12-level curriculum: standing → slow walk → omnidirectional → heavy load (2 kg) →
extreme perturbation (80 N push) → walk/stop transitions → variable-height squat.
Frozen arm policy provides continuous perturbation (random reaching + payload forces).

### Stage 2 Arm (Reaching)

```bash
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\29dof\train_unified_stage_2_arm.py ^
    --stage1_checkpoint <v6.2_model.pt> ^
    --num_envs 4096 --headless
```

## Project Structure

```
high_low_hierarchical_g1/
├── config/
│   ├── joint_config.py          # Joint names, defaults, action scales
│   └── skill_config.py          # Skill primitive parameters
├── envs/
│   └── hierarchical_env.py      # Main env: policy cascade, magnetic grasp, mode switching
├── planning/
│   ├── skill_executor.py        # Walk controller (Pure Pursuit), reach, grasp, lift, lower, place
│   ├── vlm_planner.py           # Task decomposition (simple rule-based + VLM)
│   └── semantic_map.py          # Ground-truth object/surface positions
├── low_level/
│   ├── policy_wrapper.py        # Loco policy inference wrapper
│   ├── arm_policy_wrapper.py    # Arm policy inference wrapper
│   ├── arm_controller.py        # Heuristic arm interpolation controller
│   ├── finger_controller.py     # DEX3 finger open/close
│   └── velocity_command.py      # Velocity command interface
├── perception/
│   └── perception-module-python/ # Florence-2 detector + SAM2 segmentor + DepthAnything
├── scripts/
│   ├── demo_vlm_planning.py     # End-to-end demo with video recording
│   ├── test_skills.py           # Individual skill testing
│   └── test_hierarchical.py     # Full system integration test
└── skills/
    ├── base_skill.py            # Skill base class
    ├── walk_to.py               # Walk-to-target skill
    ├── turn_to.py               # Turn-to-heading skill
    ├── stand_still.py           # Stand-still skill
    └── heuristic_manipulation.py # Grasp/place heuristics
```

## Walk Controller: Pure Pursuit

The walk controller uses **Pure Pursuit** with two modes:

| Mode | Use Case | vx | vy | vyaw | Decel Radius |
|------|----------|----|----|------|-------------|
| **Normal** | Walk to object | 0.40 | 0.20 | 0.35 | 0.5 m |
| **Carry** | Forward carry walk | 0.30 | 0.40 | 0.25 | 0.3 m |
| **Lateral** | Sideways carry to basket | 0.15 | 0.40 | 0.00 | 0.25 m |

Lateral mode maintains robot heading (vyaw = 0) and uses pure strafe. Pre-walk yaw
correction is skipped for lateral-only walks.

## Environment

- **Simulation**: Isaac Lab / Isaac Sim 5.1.0 + PhysX
- **Robot**: Unitree G1 29-DoF (12 leg + 3 waist + 7 arm + 7 finger)
- **Control frequency**: 50 Hz (4x decimation at 200 Hz physics)
- **Objects**: Steering wheel on table, basket as placement target
- **Grasp**: Magnetic attachment at 0.21 m distance

## References

- Ahn et al. 2022 -- SayCan: VLM + affordance scoring for robot task planning
- Ouyang et al. 2024 -- Berkeley Loco-Manipulation: skill chaining + VLM cascade
- Gu et al. 2025 (RSS) -- HOMIE: height-coupled knee reward for humanoid locomotion
- unitree_rl_lab -- G1 29-DoF locomotion training framework
