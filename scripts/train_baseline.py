"""Train the standard Safety Gymnasium baselines (OmniSafe PPO / PPO-Lag) on the Microduck.

Uses OmniSafe's stock config for the chosen algorithm, the same config behind its published
Safety Gymnasium benchmarks for Point, Car, etc. Only what you pass on the command line is
overridden, so a Microduck run and a Point run differ only in --env:

    python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1-v0
    python scripts/train_baseline.py --algo PPOLag --env SafetyPointGoal1-v0      # reference

How this maps onto the project's CMDP (proposal Section 2):

- state s_t: the 60-D Goal1 observation; action a_t in [-1, 1]^2 -> (v, omega) = (0.4 a1, 1.0 a2);
  reward (D_{t-1} - D_t) + 1[goal reached]; cost c_t = 1[inside any hazard] (Safety Gymnasium's
  default indicator cost); episodes end at 1,000 steps or on a fall. All of this is the env.
- PPOLag's multiplier update uses the undiscounted mean episode cost (Metrics/EpCost) against
  cost_limit = 25, i.e. J_c(pi) <= d = 25 exactly as in the formulation. The policy's advantage
  estimates use OmniSafe's gamma = 0.99 (reward and cost), as in its benchmarks.
- In the project plan these are reference baselines: PPO is the unconstrained learned policy
  and PPO-Lag the deep-RL primal-dual method (PPO primal step, gradient-ascent dual step). The
  main primal-dual method (M2) uses random search on a linear policy instead.

Train/deploy design (proposal Table 2): train with --env on the nominal robot (or a mismatched
one), then deploy the checkpoints elsewhere with evaluate_baseline.py / capture_training.py
--env, e.g.

    python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1-v0 --total-steps 2000000
    python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1MismatchSevere-v0 --total-steps 2000000
    python scripts/train_baseline.py --algo PPOLag --env SafetyMicroduckGoal1MismatchBatteryMix-v0 --total-steps 2000000
    python scripts/capture_training.py <run dir> --env SafetyMicroduckGoal1MismatchSevere-v0

The BatteryMix run is the "standard" Q3 variant (one multiplier on the mean cost over 7.4 V and
6.0 V episodes); the robust variant (one multiplier per strength) is not an OmniSafe algorithm.

Stock settings: 1 env, 20,000 steps per epoch, 10M total steps, cost limit 25 (Lagrangian
methods), 64x64 tanh actor/critic. The proposal's runs use --total-steps 2000000 and seeds 0-2.
End to end the duck trains at roughly 65 env steps/s on this laptop (plugged in), so 2M steps
is about 8-9 h. Logs and checkpoints go to runs/<algo>-{<env>}/seed-XXX-<time>/.
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
