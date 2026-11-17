"""Train the standard Safety Gymnasium baselines (OmniSafe PPO / PPO-Lag) on the Microduck.

Uses OmniSafe's stock config for the chosen algorithm, the same config behind its published
Safety Gymnasium benchmarks for Point, Car, etc. Only what you pass on the command line is
overridden, so a Microduck run and a Point run differ only in --env:

    python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1-v0
    python scripts/train_baseline.py --algo PPOLag --env SafetyPointGoal1-v0      # reference

Stock settings: 1 env, 20,000 steps per epoch, 10M total steps, cost limit 25 (Lagrangian
methods), 64x64 tanh actor/critic. The Microduck env runs ~270 steps/s on one CPU core (Point:
~700), so the default 10M steps takes roughly 12+ hours; use --total-steps for shorter runs.
Logs and checkpoints go to runs/<algo>-{<env>}/seed-XXX-<time>/.
"""

from __future__ import annotations

import argparse

import microduck_safety_gym.omnisafe_env  # noqa: F401  (must precede omnisafe)
import omnisafe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--algo', default='PPOLag', help='any OmniSafe algo, e.g. PPO, PPOLag, CPO')
    parser.add_argument('--env', default='SafetyMicroduckGoal1-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total-steps', type=int, default=None, help='default: OmniSafe (10M)')
    parser.add_argument('--steps-per-epoch', type=int, default=None, help='default: OmniSafe (20k)')
    parser.add_argument('--cost-limit', type=float, default=None, help='default: OmniSafe (25)')
    parser.add_argument('--torch-threads', type=int, default=None)
    parser.add_argument(
        '--device',
        default=None,
        help="'cpu' or 'cuda:0' for the policy/critic networks (default: OmniSafe, cpu). "
        'The env always runs on CPU.',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=5,
        help='checkpoint every N epochs (OmniSafe default: 100). Only affects saving, not '
        'learning; scripts/capture_training.py rolls out each checkpoint.',
    )
    parser.add_argument('--log-dir', default='./runs')
    parser.add_argument('--use-wandb', action='store_true')
    args = parser.parse_args()

    custom_cfgs: dict = {
        'seed': args.seed,
        'train_cfgs': {},
        'algo_cfgs': {},
        'logger_cfgs': {
            'log_dir': args.log_dir,
            'use_wandb': args.use_wandb,
            'save_model_freq': args.save_every,
        },
    }
    if args.total_steps is not None:
        custom_cfgs['train_cfgs']['total_steps'] = args.total_steps
    if args.torch_threads is not None:
        custom_cfgs['train_cfgs']['torch_threads'] = args.torch_threads
    if args.device is not None:
        custom_cfgs['train_cfgs']['device'] = args.device
    if args.steps_per_epoch is not None:
        custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch
    if args.cost_limit is not None:
        custom_cfgs['lagrange_cfgs'] = {'cost_limit': args.cost_limit}

    agent = omnisafe.Agent(args.algo, args.env, custom_cfgs=custom_cfgs)
    agent.learn()


if __name__ == '__main__':
    main()
