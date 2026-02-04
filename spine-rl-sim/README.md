# Spine RL Simulator (MuJoCo)

This is a **research simulator** scaffold based on your `sub-verse563` MuJoCo mesh XML. It creates one rigid body per vertebra, applies random "fractures" (pose offsets) to multiple vertebrae at reset, and trains an agent to restore those poses.

**Important**: This is not a clinical tool. It is a simplified, kinematic learning environment that can be used for RL research/prototyping.

## What it does
- Loads your spine mesh assets from `sub-verse563_ALL_gt_ts.xml`.
- Builds a new MuJoCo model with a free joint for each vertebra.
- Randomly offsets multiple vertebrae (translation + rotation) at each episode reset.
- Defines a Gymnasium environment where actions nudge fractured vertebrae back to the target pose.
- Adds basic safety and anatomical regularization (smoothness + adjacency penalties).
- Logs training/validation metrics and plots them.

## Setup
1) Generate the model and copy meshes:
```bash
python3 scripts/build_mujoco_model.py
```
This writes `assets/spine_model.xml` and copies meshes into `assets/meshes/`.

2) Install requirements:
```bash
pip install -r requirements.txt
```

3) Run a simple random policy (rendering optional):
```bash
python3 run_env.py
```

## Training + Metrics
Train PPO with periodic evaluation:
```bash
python3 train_ppo.py
```
This saves:
- `metrics/train_metrics.csv`
- `metrics/eval_metrics.csv`
- `models/ppo_spine_fix`

Plot training vs validation curves:
```bash
python3 plot_metrics.py
```
This outputs `metrics/metrics_plot.png` with a legend at the top right.

## Configuration
The environment is controlled by `SpineFixConfig` in `spine_rl/envs/spine_fix_env.py`:
- `num_fractures`: number of vertebrae to fracture per episode
- `fracture_translation`, `fracture_rotation_deg`: fracture severity
- `action_translation`, `action_rotation_deg`: per-step action limits
- `pos_threshold`, `rot_threshold_deg`: success criteria
- `smoothness_weight`: penalizes jerky motion
- `adjacency_weight`: penalizes separation of neighboring vertebrae
- `max_pos_radius`, `max_rot_deg`: safety bounds (terminates episode)

## Notes / Limitations
- This is **kinematic** control (directly editing pose) rather than surgical tool dynamics.
- There are no collisions or soft tissue dynamics.
- The target is the original pose; no anatomical constraints are enforced beyond simple penalties.

If you want, I can extend this into a more realistic surgical manipulation environment (instruments, contact forces, constraints, multi‑vertebra alignment, etc.).
