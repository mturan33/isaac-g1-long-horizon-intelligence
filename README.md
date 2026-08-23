# Hierarchical VLM-RL Pick-and-Place & Drawer Opening for G1 Humanoid

End-to-end autonomous manipulation on the Unitree G1 (29 DoF) in NVIDIA Isaac Lab.
A VLM (Qwen3-VL) decomposes natural language tasks into skill primitives, executed by
a triple-policy cascade (locomotion + arm + finger) trained entirely in simulation.

Supports two task types:
- **Pick-and-place**: Walk to object, grasp, carry, place in basket
- **Drawer opening**: Walk to cabinet, grasp handle, pull drawer open

<img src="https://github.com/user-attachments/assets/700bcdf9-4e1d-447d-8534-e7c3d9fe6bb6" width="600"/>
<img src="https://github.com/user-attachments/assets/a76af9e9-a317-438d-ba78-8aa116ef7d87" width="600"/>

## Key Results

Validated on Apr 2026 with the policies shipped in `checkpoints/`
(Stage 2 Loco + Stage 2 Arm trained on RTX 5070 Ti Laptop, 64 GB DDR5):

| Task | Setup | walk_to success | Standing | Notes |
|------|-------|-----------------|----------|-------|
| Pick-and-place (rule) | 1 env GUI | 1/1 | 8/8 | full pre_reach + walk_to + reach pipeline |
| Pick-and-place (rule) | 8 envs headless | **4/7 (majority)** | 7/8 | per-env vectorized PP, 1 fall |
| Drawer opening (rule) | 1 env GUI | not re-measured | — | (planner unchanged from earlier release) |

- **Zero falls** in 1-env runs; 7/8 standing in 8-env headless runs
- **VLM closed-loop**: Background replanning every ~10s (planner unchanged)
- Lateral carry with heading-hold Pure Pursuit controller
- Physical drawer pull via arm retraction + backward walk

> **walk_to success rate**: 4/7+ in headless multi-env, 1/1 reliable in single-env
> GUI demo. Subsequent skills (`reach` / `grasp`) are scripted state machines and
> are being replaced with a learned VLA policy in a separate repo.

## Architecture

```
  "Open the drawer"          VLM Planner (Qwen3-VL 4B)
         |                          |
         v                          v
  +-----------------+    +-------------------+
  | Semantic Map    |--->| Skill Plan (JSON) |
  | (ground truth)  |    | pre_reach, walk,  |
  +-----------------+    | reach, grasp,     |
                         | pull, release     |
                         +--------+----------+
                                  |
                    +-------------v--------------+
                    |      Skill Executor         |
                    |  Pure Pursuit walk + PID    |
                    +------+-------------+-------+
                           |             |
                   +-------v------+ +----v-------+
                   | Loco Policy  | | Arm Policy  |
                   | 66->15 (50Hz)| | 39->7 (50Hz)|
                   +-------+------+ +----+-------+
                           |             |
                   +-------v-------------v-------+
                   |   Isaac Lab / PhysX (50 Hz)  |
                   |   G1 29-DoF + Cabinet + Table|
                   +-----------------------------+
```

## Quick Start

### Prerequisites

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| GPU | NVIDIA RTX (Blackwell: use driver 591.74) |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.48.0 (release/2.3.0) |
| Ollama | Latest (for VLM planner) |

### Installation

```bash
# 1. Conda environment
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# 2. Isaac Sim
pip install isaacsim==5.1.0 isaacsim-kernel==5.1.0 isaacsim-core==5.1.0 \
    --extra-index-url https://pypi.nvidia.com

# 3. Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git C:\IsaacLab
cd C:\IsaacLab
git checkout release/2.3.0
.\isaaclab.bat --install

# 4. Fix dependencies
pip install h5py==3.11.0 --force-reinstall --no-cache-dir
pip install numpy==1.26.0

# 5. VLM planner (optional)
pip install ollama
# Install Ollama app from https://ollama.com, then:
ollama pull qwen3-vl:4b

# 6. Copy this project into IsaacLab source tree
# Place high_low_hierarchical_g1/ under:
# C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\
```

### Scene Assets (required)

The scene USDs (table, cabinet, and the Dex3-handed G1) are **not** part of this
repository. They are published by Unitree in
[`unitree_sim_isaaclab`](https://github.com/unitreerobotics/unitree_sim_isaaclab)
(Apache-2.0) and distributed as a ~1.2 GB Git-LFS archive, so they are fetched
separately:

```bash
cd C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
cd unitree_sim_isaaclab && bash fetch_assets.sh
```

`fetch_assets.sh` clones the
[HF dataset](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds),
unzips `assets.zip`, and moves `assets/` into the repository root. It is a bash
script and uses `unzip`, so on Windows run it from **Git Bash** (or unzip
`assets.zip` manually into `unitree_sim_isaaclab/assets/`).

`unitree_sim_isaaclab/` must sit **next to** this project, not inside it:

```
direct/
├── high_low_hierarchical_g1/     <- this repository
└── unitree_sim_isaaclab/         <- cloned above
    └── assets/
        ├── objects/PackingTable/PackingTable.usd
        ├── objects/drawers/cabinet_collider.usd
        └── robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd
```

Three files are loaded by `envs/hierarchical_env.py`. Verified against the
upstream archive (`assets.zip`, `sha256:06fbf145…ead4b0`) — unmodified:

| File | Size | SHA-256 (first 16) |
|------|------|--------------------|
| `objects/PackingTable/PackingTable.usd` | 16.5 MB | `841457256b476a7d` |
| `objects/drawers/cabinet_collider.usd` | 267 KB | `af9703419973d930` |
| `robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd` | 40.6 MB | `01677e6ab1d321e7` |

> **Note:** these paths are currently hard-coded as absolute Windows paths in
> `envs/hierarchical_env.py` (lines 74, 326, 362). If your Isaac Lab checkout is
> not at `C:\IsaacLab`, edit those three lines to match your layout.

### Pre-trained Checkpoints

Pre-trained policies (trained on this same hardware/config) are shipped under `checkpoints/`:

| File | Stage | Size |
|------|-------|------|
| `checkpoints/loco_stage2.pt` | Stage 2 Loco (perturbation-robust) | 5.1 MB |
| `checkpoints/arm_stage2.pt` | Stage 2 Arm (3 cm reach accuracy) | 4.1 MB |

No training required — clone and run demos directly.

### Run Demos

All commands from `C:\IsaacLab`:

```bash
# === PICK-AND-PLACE ===

# Simple planner (no VLM, instant)
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py ^
    --num_envs 1 ^
    --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\loco_stage2.pt ^
    --arm_checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\arm_stage2.pt ^
    --task "Pick up the steering wheel from the table" ^
    --planner simple

# VLM planner (requires Ollama running)
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py ^
    --num_envs 1 ^
    --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\loco_stage2.pt ^
    --arm_checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\arm_stage2.pt ^
    --task "Pick up the steering wheel from the table" ^
    --planner vlm --vlm_model qwen3-vl:4b

# === DRAWER OPENING ===

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py ^
    --num_envs 1 ^
    --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\loco_stage2.pt ^
    --arm_checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\arm_stage2.pt ^
    --task "Open the drawer" ^
    --planner vlm --vlm_model qwen3-vl:4b

# === CLOSED-LOOP (VLM replans every ~10s) ===

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py ^
    --num_envs 1 ^
    --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\loco_stage2.pt ^
    --arm_checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\checkpoints\arm_stage2.pt ^
    --task "Open the drawer" ^
    --planner vlm --vlm_model qwen3-vl:4b --closed_loop

# Add --headless for no GUI (faster)
# Add --record for video capture
```

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--planner simple` | Rule-based planner (no VLM needed) |
| `--planner vlm` | VLM planner via Ollama |
| `--vlm_model qwen3-vl:4b` | Ollama model name |
| `--closed_loop` | Enable VLM background replanning |
| `--headless` | No GUI (faster, for testing) |
| `--record` | Record video frames |
| `--num_envs N` | Number of parallel environments |

## Skill Library

### Pick-and-Place Skills

| # | Skill | Description |
|---|-------|-------------|
| 1 | `pre_reach` | Raise arm high before approaching table |
| 2 | `walk_to` | Pure Pursuit walk to target (object/surface) |
| 3 | `reach` | Arm policy extends to object |
| 4 | `grasp` | Magnetic grasp + finger close |
| 5 | `lift` | Lift object above table height |
| 6 | `walk_to` | Lateral carry walk to basket |
| 7 | `lower` | Lower arm to basket level |
| 8 | `place` | Release object, return arm |

### Drawer Skills

| # | Skill | Description |
|---|-------|-------------|
| 1 | `pre_reach` | Raise arm high |
| 2 | `walk_to` | Walk to drawer handle |
| 3 | `reach` | Hold arm for grasp |
| 4 | `grasp` | Magnetic attach to handle |
| 5 | `pull` | Arm retraction + backward walk |
| 6 | `release` | Detach, return arm to default |

## VLM Planner

The VLM (Qwen3-VL via Ollama) receives:
- World state JSON (robot pos, objects, surfaces, interactables)
- Task description in natural language

And outputs a JSON skill plan:
```json
{"plan": [
  {"skill": "pre_reach", "params": {"target": "drawer_01"}},
  {"skill": "walk_to", "params": {"target": "drawer_01", "stop_distance": 0.8, "hold_arm": true}},
  {"skill": "pull", "params": {"direction": [-1, 0, 0], "distance": 0.25}},
  {"skill": "release", "params": {}}
]}
```

### Closed-Loop Mode

With `--closed_loop`, a background thread continuously:
1. Updates the semantic map
2. Calls VLM for replanning decisions
3. Can modify the plan mid-execution if conditions change

## Walk Controller: Pure Pursuit

| Mode | Use Case | vx | vy | vyaw |
|------|----------|----|----|------|
| **Normal** | Walk to object | 0.40 | 0.20 | 0.35 |
| **Carry** | Forward carry | 0.30 | 0.40 | 0.25 |
| **Lateral** | Sideways to basket | 0.15 | 0.40 | hold |

Lateral mode uses heading-hold P-controller (Kp=2.5) to maintain orientation.

## Scene

- **Robot**: Unitree G1 29-DoF (12 leg + 3 waist + 7 arm + 7 finger)
- **Table**: PackingTable with basket
- **Object**: Steering wheel (scaled 0.75x)
- **Cabinet**: Sektion cabinet with prismatic drawer joints (scaled 1.3x)
- **Control**: 50 Hz (4x decimation at 200 Hz physics)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GUI crash on Blackwell GPU | Use NVIDIA driver 591.74 (not 595.x) |
| h5py DLL error | `pip install h5py==3.11.0` |
| numpy conflict | `pip install numpy==1.26.0` |
| VLM returns empty | Ensure Ollama is running (`ollama serve`) |
| "filename too long" DLL errors | Non-blocking; enable Windows long paths |
| Camera lock in GUI | Only locks with `--record`; free otherwise |

## Scoring System

VLM runs are auto-scored (0-10) and saved to `results/vlm_runs/`:

| Criterion | Points |
|-----------|--------|
| Step completion (N/total) | 5.0 |
| Robot standing at end | 2.0 |
| Speed (< 60s = 2, < 120s = 1) | 2.0 |
| VLM plan used (not fallback) | 1.0 |

## References

- Ahn et al. 2022 -- SayCan: VLM + affordance scoring
- Ouyang et al. 2024 -- Berkeley Loco-Manipulation
- Gu et al. 2025 (RSS) -- HOMIE: height-coupled knee reward
- Coulter 1992 -- Pure Pursuit path tracking
- unitree_rl_lab -- G1 29-DoF locomotion framework
