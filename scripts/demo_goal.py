"""Scripted demo: run B0 (goal-lidar baseline) and optionally M3 (CBF safety filter).

    python scripts/demo_goal.py --episodes 3
    python scripts/demo_goal.py --episodes 1 --cbf --video runs/videos/demo_cbf.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import microduck_safety_gym  # noqa: F401  (registers the Microduck envs)
import safety_gymnasium

from microduck_safety_gym.controllers import CBFSafetyFilter, LidarGoalPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--env', default='SafetyMicroduckGoal1-v0')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cbf', action='store_true', help='apply M3 CBF safety filter on top of B0')
    parser.add_argument('--k-psi', type=float, default=2.0)
    parser.add_argument('--psi-0', type=float, default=0.6)
    parser.add_argument('--cbf-lookahead', type=float, default=0.15)
    parser.add_argument('--cbf-alpha', type=float, default=1.0)
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
    policy = LidarGoalPolicy.from_env(env, k_psi=args.k_psi, psi_0=args.psi_0)
    cbf_filter = CBFSafetyFilter(lookahead=args.cbf_lookahead, alpha=args.cbf_alpha) if args.cbf else None

    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        task = env.unwrapped.task
        ret = cost = goals = steps = overrides = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(obs)
            if cbf_filter is not None:
                action, cbf_info = cbf_filter.filter(task, action)
                overrides += int(bool(cbf_info['cbf_overridden']))
            obs, r, c, terminated, truncated, info = env.step(action)
            ret, cost, steps = ret + r, cost + c, steps + 1
            goals += int(info.get('goal_met', False))
            if render and steps % 2 == 0:  # 25 fps
                frames.append(env.render())
        outcome = 'fell' if terminated else 'time limit'
        ovr_pct = 100.0 * overrides / steps if steps else 0.0
        print(
            f'episode {ep}: return {ret:6.2f} | cost {cost:4.0f} | goals {goals} | '
            f'override {ovr_pct:4.1f}% | {steps} steps ({outcome})'
        )
    env.close()

    if render:
        import imageio.v2 as imageio  # noqa: PLC0415

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=25)
        print(f'wrote {args.video} ({len(frames)} frames)')


if __name__ == '__main__':
    main()
