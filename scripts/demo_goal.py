"""Scripted demo: walk the Microduck to Safety Gymnasium goals with the pretrained gait.

A hand-written high-level controller, not a learned one. It reads the goal position (and,
with --avoid-hazards, the hazard positions) from the simulator, turns toward the goal and
walks. It exists to show that the duck can move around the Goal1 arena and reach goals through
the same 2-D (forward, turn) action an RL agent uses.

    python scripts/demo_goal.py --episodes 3
    python scripts/demo_goal.py --episodes 1 --avoid-hazards --video runs/videos/demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import microduck_safety_gym  # noqa: F401  (registers the Microduck envs)
import safety_gymnasium


def heading_error(task, target_xy: np.ndarray) -> float:
    """Angle from the duck's forward axis to ``target_xy``, in (-pi, pi]."""
    rel = target_xy - task.agent.pos[:2]
    forward = task.agent.mat[:2, 0]
    return float(np.arctan2(forward[0] * rel[1] - forward[1] * rel[0], forward @ rel))


def steering_target(task, avoid_hazards: bool) -> np.ndarray:
    """Goal position, nudged away from nearby hazards when ``avoid_hazards``."""
    pos = task.agent.pos[:2]
    goal = task.goal.pos[:2]
    if not avoid_hazards or not hasattr(task, 'hazards'):
        return goal
    to_goal = goal - pos
    direction = to_goal / (np.linalg.norm(to_goal) + 1e-9)
    clearance = task.hazards.size + 0.15
    for hazard in task.hazards.pos:
        away = pos - hazard[:2]
        dist = np.linalg.norm(away)
        if dist < clearance + 0.35:
            direction += 1.5 * (clearance + 0.35 - dist) / 0.35 * away / (dist + 1e-9)
    return pos + direction


def policy(task, avoid_hazards: bool) -> np.ndarray:
    """(forward, turn) in [-1, 1]: turn toward the target, walk when roughly facing it."""
    err = heading_error(task, steering_target(task, avoid_hazards))
    # The walking policy ignores small yaw commands (Pollen documents a dead zone below
    # ~0.5 rad/s in place), so turn at >= 0.8 of the max rate when the error is large.
    turn = np.clip(2.0 * err, -1.0, 1.0)
    if abs(err) > 0.35:
        turn = np.sign(err) * max(abs(turn), 0.8)
    forward = 1.0 if abs(err) < 0.6 else 0.0
    return np.array([forward, turn])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyMicroduckGoal1-v0')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--avoid-hazards', action='store_true')
    parser.add_argument('--video', type=Path, default=None, help='write an mp4 of the run')
    parser.add_argument('--camera', default='fixednear', help='fixednear, fixedfar, track, vision')
    args = parser.parse_args()

    render = args.video is not None
    env = safety_gymnasium.make(
        args.env,
        render_mode='rgb_array' if render else None,
        camera_name=args.camera,
        width=640,
        height=480,
    )
    frames = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        task = env.unwrapped.task
        ret = cost = goals = steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, r, c, terminated, truncated, info = env.step(policy(task, args.avoid_hazards))
            ret, cost, steps = ret + r, cost + c, steps + 1
            goals += int(info.get('goal_met', False))
            if render and steps % 2 == 0:  # 25 fps
                frames.append(env.render())
        outcome = 'fell' if terminated else 'time limit'
        print(
            f'episode {ep}: return {ret:6.2f} | cost {cost:4.0f} | goals {goals} | '
            f'{steps} steps ({outcome})',
        )
    env.close()

    if render:
        import imageio.v2 as imageio  # noqa: PLC0415

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=25)
        print(f'wrote {args.video} ({len(frames)} frames)')


if __name__ == '__main__':
    main()
