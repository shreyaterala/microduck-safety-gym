"""Evaluate a trained OmniSafe checkpoint on its env, optionally recording a video.

    python scripts/evaluate_baseline.py runs/PPOLag-{SafetyMicroduckGoal1-v0}/seed-000-<time>
    python scripts/evaluate_baseline.py <run dir> --model epoch-500.pt --episodes 10 --video
    python scripts/evaluate_baseline.py <run dir> --env SafetyMicroduckGoal1MismatchSevere-v0

Uses ``omnisafe.Evaluator``, which rebuilds the env from the run's config.json and runs the
deterministic actor. ``--env`` deploys the policy on a different env instead (train on the
nominal robot, evaluate on a mismatched one). Prints mean episode return and cost; --video writes mp4s under the run's
``video/`` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microduck_safety_gym.omnisafe_env import load_evaluator  # (must precede omnisafe)


def latest_checkpoint(run_dir: Path) -> str:
    saves = sorted(
        (run_dir / 'torch_save').glob('epoch-*.pt'),
        key=lambda p: int(p.stem.split('-')[1]),
    )
    if not saves:
        raise FileNotFoundError(f'no torch_save/epoch-*.pt under {run_dir}')
    return saves[-1].name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--model', default=None, help='checkpoint name (default: latest epoch)')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--env', default=None, help='deploy on this env id (default: the training env)')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--camera', default='fixednear')
    args = parser.parse_args()

    model = args.model or latest_checkpoint(args.run_dir)
    evaluator = load_evaluator(
        args.run_dir, model, env_id=args.env, camera_name=args.camera, width=640, height=480
    )
    print(f'{args.run_dir.name} / {model} on {evaluator._cfgs["env_id"]}')  # noqa: SLF001
    evaluator.evaluate(num_episodes=args.episodes)
    if args.video:
        evaluator.render(num_episodes=1)


if __name__ == '__main__':
    main()
