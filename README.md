# microduck-safety-gym

[Pollen Robotics' Microduck](https://pollen-robotics.com/microduck/) (25 cm, 14-servo biped) as a
[Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) robot, next to Point, Car,
Doggo, Racecar and Ant. The duck walks with Pollen's pretrained walking policy; an RL agent steers
it with the same 2-D action Point uses, on the unchanged Goal tasks.

```python
import microduck_safety_gym  # registers SafetyMicroduck{GoalBase,Goal0,Goal1,Goal2}-v0
import safety_gymnasium

env = safety_gymnasium.make('SafetyMicroduckGoal1-v0')
obs, info = env.reset(seed=0)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
```

Import `microduck_safety_gym` **before** `safety_gymnasium` or `omnisafe` (see [Compatibility](#compatibility)).

## How the duck plugs in

```
RL policy ──(forward, turn) ∈ [-1,1]²──▶ command (vx, 0, yaw rate)
   20 ms                                   │ slew-limited
                                           ▼
                  velstand.onnx walking policy (50 Hz, 61-D obs → 14 joint targets)
                                           ▼
                  BAM model of the XL330 servos ──▶ MuJoCo, 4 × 5 ms steps
```

| | Point (stock) | Microduck |
|---|---|---|
| Action | 2-D in [-1, 1]: force, turn | 2-D in [-1, 1]: forward speed (×0.4 m/s), yaw rate (×1.0 rad/s) |
| Observation (Goal1) | 60-D: accelerometer, velocimeter, gyro, magnetometer + goal/hazard/vase lidar | same 60-D layout; sensors sit on the duck's trunk |
| Control period | 10 × 2 ms = 20 ms | 4 × 5 ms = 20 ms (one walking-policy tick) |
| Episode | 1,000 steps (20 s) | 1,000 steps (20 s) |
| Task, reward, cost, layout | Goal1: 8 hazards, 1 vase, extents ±1.5 m | identical (unmodified Safety Gymnasium task) |
| Termination | never | on a fall (trunk < 6 cm or tilt > 60°), as Ant does |

Two duck-specific details: the command is **slew-rate limited** (1 m/s², 4 rad/s²) because the walking
policy was trained on piecewise-constant commands, and each episode starts with **0.5 s of standing**
so it begins from a settled stance.

### What the walking policy can do

`scripts/check_gait.py` measures it in a plain scene (4 s at each constant command):

| Command | Achieved |
|---|---|
| forward 0.1 / 0.2 / 0.25 / 0.3 / 0.35 / 0.4 m/s | 0.00 / 0.00 / 0.07 / 0.09 / 0.11 / 0.13 m/s |
| yaw drift while walking straight at 0.4 m/s | 0.11 rad/s |
| turn in place 0.5 / 1.0 rad/s | 0.00 / 1.03 rad/s |

The duck is slow (about 0.13 m/s flat out, against roughly an order of magnitude more for Point) and
ignores small commands. Pollen documents the same behavior for this policy in their own sim
(0.4 m/s → 0.17 achieved; in-place turns below ~0.6 rad/s ≈ 0;
`third_party/microduck_rl/docs/velstand_policy.md`). It is part of the embodiment, so the Goal task
is left unchanged rather than rescaled.

## Setup

Python 3.12 (required by the BAM actuator package) with [uv](https://docs.astral.sh/uv/). The env
is kept outside OneDrive so syncing does not fight the install:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/microduck-safety-gym"   # PowerShell: $env:UV_PROJECT_ENVIRONMENT=...
uv sync
uv run python scripts/fetch_assets.py   # clones microduck_rl, downloads velstand.onnx
uv run python -m pytest tests -q
```

Assets are fetched on first use anyway and are not committed:

- `third_party/microduck_rl` — [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
  at commit `bae39f5`; the env uses `robot_allcollisions.xml` (MJCF Apache-2.0, meshes CC BY-NC-SA 4.0).
- `assets/policies/velstand.onnx` — [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies),
  the shipped default walk (walks, stands at zero command, gets up after falls).

## Scripts

| Script | What it does |
|---|---|
| `scripts/demo_goal.py` | B0 goal-lidar baseline (`--cbf` for M3 CBF safety filter, `--video out.mp4`) |
| `scripts/train_primal_dual.py` | M2 primal-dual random search on linear policy (`single`, `q3-standard`, `q3-robust`, optional `--cbf`) |
| `scripts/evaluate_experiments.py` | 100-layout evaluation for B0, B0+M3, M2, and M2+M3 across all mismatch levels and single-knob ablations |
| `scripts/train_baseline.py` | Optional OmniSafe PPO / PPO-Lag reference baselines with stock Safety Gymnasium config |
| `scripts/evaluate_baseline.py` | Score an OmniSafe checkpoint (`omnisafe.Evaluator`), optional video |
| `scripts/capture_training.py` | Roll out every checkpoint of an OmniSafe run: labeled videos, eval CSV, training curves, progression video |
| `scripts/check_gait.py` | Walking-policy speed and turn tracking, outside Safety Gymnasium |
| `scripts/check_terrain.py` | Walking policy (no retraining) on low friction, bumps, slopes, ledges and steps: speed, drift, falls, recovery |

### Scripted demo (B0 and B0+M3)

```bash
uv run python scripts/demo_goal.py --episodes 3
uv run python scripts/demo_goal.py --episodes 1 --cbf --video runs/videos/demo_cbf.mp4
```

### Main experiments (M2 primal-dual & 100-layout evaluation)

```bash
# Q2: evaluate B0 and B0+M3 (CBF) on 100 fixed layouts across all levels & single-knob ablations
uv run python scripts/evaluate_experiments.py --method b0 --episodes 100
uv run python scripts/evaluate_experiments.py --method b0+cbf --episodes 100

# Q1: train M2 (122-param linear policy, random search) on nominal and severe mismatch
uv run python scripts/train_primal_dual.py --mode single --env SafetyMicroduckGoal1-v0 --seed 0
uv run python scripts/train_primal_dual.py --mode single --env SafetyMicroduckGoal1MismatchSevere-v0 --seed 0

# Q3: standard (single lambda) vs. robust (one lambda per motor strength) across 7.4 V and 6.0 V
uv run python scripts/train_primal_dual.py --mode q3-standard --seed 0
uv run python scripts/train_primal_dual.py --mode q3-robust --seed 0
```

### RL baselines

The baseline is OmniSafe with its stock config for each algorithm, the same setup as its
Point/Car Safety Gymnasium benchmarks. Only the env id changes:

```bash
uv run python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1-v0
uv run python scripts/train_baseline.py --algo PPO    --env SafetyMicroduckGoal1-v0
uv run python scripts/train_baseline.py --algo PPOLag --env SafetyPointGoal1-v0      # reference
uv run python scripts/evaluate_baseline.py "runs/PPOLag-{SafetyMicroduckGoal1-v0}/seed-000-<time>" --video
```

Stock settings: 1 env, 20k steps per epoch, 10M total steps, cost limit 25, 64×64 networks. The one
non-stock default is `--save-every 5` (a checkpoint every 100k steps instead of OmniSafe's every 100
epochs); it changes what is saved, not what is learned.

**How they fit the project's CMDP.** The env supplies the formulation unchanged: 60-D state,
(forward, turn) action scaled to (0.4 m/s, 1.0 rad/s), reward = progress + goal bonus, cost =
1 if inside any hazard (Safety Gymnasium's indicator cost), episodes of 1,000 steps that also end
on a fall. PPO-Lag's multiplier update compares the undiscounted mean episode cost with
`cost_limit` = 25, which is the constraint J_c(π) ≤ d = 25; the advantage estimates use OmniSafe's
γ = 0.99. In the project these are reference baselines: PPO is the unconstrained learned policy
and PPO-Lag the deep-RL primal-dual method; the main primal-dual method trains a linear policy
by random search instead.

**Train / deploy.** Train on one env and deploy the checkpoints on another with `--env`
(evaluated with the training run's observation normalizer):

```bash
# train on the nominal robot, then deploy on mismatched ones (Q1)
uv run python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1-v0 --total-steps 2000000 --seed 0
uv run python scripts/evaluate_baseline.py "<run dir>" --env SafetyMicroduckGoal1MismatchSevere-v0
uv run python scripts/capture_training.py "<run dir>" --env SafetyMicroduckGoal1MismatchModerate-v0   # -> <run>/capture-<env>/
# train under severe mismatch (Q1, training-time violations)
uv run python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1MismatchSevere-v0 --total-steps 2000000
# Q3 "standard" variant: one multiplier on the mean cost of 7.4 V and 6.0 V episodes
uv run python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1MismatchBatteryMix-v0 --total-steps 2000000
```

The robust Q3 variant (one multiplier per motor strength) is not an OmniSafe algorithm. Runs
from before the reset fix (bind before setting the standing pose, Nov 2026) started each episode
from a collapsed pose; retrain rather than compare against them.

### Capturing training

`capture_training.py` turns a run into something you can watch. For each checkpoint it rolls out
the deterministic policy on fixed seeds (the same layouts at every checkpoint) and writes to
`<run>/capture/`:

| File | Contents |
|---|---|
| `epoch-N.mp4` | one 20 s episode, overlaid with epoch, env steps, return, cost, goals; red border on steps that incur cost |
| `eval.csv` | per checkpoint: mean return, cost, goals, falls over `--eval-episodes` (default 3) |
| `training_curves.png` | OmniSafe's per-epoch return, cost vs. limit and Lagrange multiplier, with checkpoint evals overlaid |
| `progression.mp4` | every checkpoint's clip (first `--clip-seconds`, default 10) behind a title card: the policy learning |

```bash
# alongside a run in progress: captures each checkpoint as it appears, exits after the last one
uv run python scripts/capture_training.py "runs/PPOLag-{SafetyMicroduckGoal1-v0}/seed-000-<time>" --watch
# or once, on a finished run
uv run python scripts/capture_training.py "runs/PPOLag-{SafetyMicroduckGoal1-v0}/seed-000-<time>"
```

Each checkpoint takes about 30 s to capture (3 episodes plus rendering), small next to the
~12 min between checkpoints.
On this laptop the duck trains at about **65 env steps/s** end to end, so 10M steps is roughly 40 h.
`--total-steps 2000000` (about 8–9 h) is a practical first run; report the step count when comparing
against published 10M-step numbers.

## Mismatch levels

`SafetyMicroduck<task>Mismatch<Level>-v0` are the same tasks with the robot following its
commands less faithfully (`microduck_safety_gym/mismatch.py`):

| Level | Command delay | Servo supply | Shoves (0.1 s, every 2-4 s) | Floor friction |
|---|---|---|---|---|
| none (`...Goal1-v0`) | 0 ms | 7.4 V | none | 1.0 |
| `Moderate` | 40 ms | 6.6 V | 2.0 N | 0.4 |
| `Severe` | 100 ms | 6.0 V | 2.5 N | 0.3 |
| `Delay` | 100 ms | 7.4 V | none | 1.0 |
| `Battery` | 0 ms | 6.0 V | none | 1.0 |
| `Shoves` | 0 ms | 7.4 V | 2.5 N | 1.0 |
| `Friction` | 0 ms | 7.4 V | none | 0.3 |
| `BatteryMix` | 0 ms | 7.4 V or 6.0 V, 50/50 per episode | none | 1.0 |

`Delay`, `Battery`, `Shoves` and `Friction` are the single-knob ablation (each knob alone at its
severe value). `Battery` is the "reduced motor strength" of Q3 and `BatteryMix` its training mix.

Shoves and friction were set just below the point where the walking policy starts falling
(see the comments in `mismatch.py`; `scripts/check_terrain.py` measures the policy alone).

## Compatibility

Safety Gymnasium 1.0.0 targets Python ≤ 3.10 and MuJoCo 2.3.3; the BAM actuator model needs
Python 3.12 and MuJoCo 3. `microduck_safety_gym._compat` bridges that:

- **dataclass defaults:** safety-gymnasium uses NumPy arrays as dataclass defaults, which Python
  ≥ 3.11 rejects. While `safety_gymnasium` is imported, those defaults become `default_factory`
  copies (the same fix people apply by hand).
- **rendering:** gymnasium 0.28.1's marker drawing sets a field MuJoCo 3 removed; it is replaced
  with the `mjv_initGeom` version newer gymnasium uses.
- `pyproject.toml` overrides a few pins to their nearest Python 3.12 builds (mujoco 3.7, numpy
  1.26, pygame 2.5+, pandas 2.2, moviepy < 2) and pins BAM to the commit microduck_rl trains with.

Because the fix runs at import, keep OmniSafe's `vector_env_nums: 1` (its default). Subprocess
vector envs would import `safety_gymnasium` before this package.

## Layout

```
microduck_safety_gym/
  __init__.py      registers the agent and SafetyMicroduck* envs
  agent.py         Microduck(BaseAgent): action scaling, slew limit, fall check, reset/settle
  builder.py       Builder whose physics loop runs the policy tick and BAM updates
  controller.py    walking policy (ONNX) + BAM actuators
  controllers.py   B0 (LidarGoalPolicy), M2 (LinearPolicy), M3 (CBFSafetyFilter + 2-D QP)
  eval_metrics.py  episode rollout and Section 5 metrics (cost, falls, stalls, incursion, tracking error)
  mismatch.py      mismatch knobs and levels (Moderate, Severe, single-knob ablations, BatteryMix)
  model.py         builds the Safety Gymnasium agent XML from Pollen's MJCF
  omnisafe_env.py  lets OmniSafe accept the Microduck env ids
  paths.py         asset locations and fetching
  _compat.py       Python 3.12 / MuJoCo 3 fixes for safety-gymnasium 1.0.0
scripts/  tests/
```
