"""Train M2: Primal-Dual with Random Search (ARS) on a 122-parameter linear policy.

Implements Proposal Section 3.2 (Q1 and Q3):
- ``--mode single``: standard M2 on a single env (e.g. nominal or MismatchSevere for Q1).
- ``--mode q3-standard``: runs each perturbation at both 7.4 V (nominal) and 6.0 V (Battery),
  updating a single Lagrange multiplier against (C_nom + C_red) / 2 <= d.
- ``--mode q3-robust``: runs each perturbation at both 7.4 V and 6.0 V, maintaining separate
  multipliers lambda_nom, lambda_red to target max(C_nom, C_red) <= d.

Examples:
    python scripts/train_primal_dual.py --mode single --env SafetyMicroduckGoal1-v0 --seed 0
    python scripts/train_primal_dual.py --mode single --env SafetyMicroduckGoal1MismatchSevere-v0 --seed 0
    python scripts/train_primal_dual.py --mode q3-standard --seed 0
    python scripts/train_primal_dual.py --mode q3-robust --seed 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from microduck_safety_gym.controllers import CBFSafetyFilter, LinearPolicy
from microduck_safety_gym.eval_metrics import rollout_episode

# Worker-local globals (initialized once per process).
_WORKER_ENVS: dict[str, object] = {}
_WORKER_CBF: CBFSafetyFilter | None = None
_WORKER_CBF_PENALTY: float = 0.0


def _init_worker(env_ids: list[str], use_cbf: bool, cbf_lookahead: float, cbf_alpha: float, cbf_penalty: float) -> None:
    global _WORKER_CBF, _WORKER_CBF_PENALTY  # noqa: PLW0603
    import microduck_safety_gym  # noqa: F401, PLC0415
    import safety_gymnasium  # noqa: PLC0415

    for eid in env_ids:
        _WORKER_ENVS[eid] = safety_gymnasium.make(eid)
    _WORKER_CBF = CBFSafetyFilter(lookahead=cbf_lookahead, alpha=cbf_alpha) if use_cbf else None
    _WORKER_CBF_PENALTY = cbf_penalty


def _eval_pair_worker(task_args: tuple[str, np.ndarray, np.ndarray, int]) -> tuple[dict, dict]:
    """Evaluate theta + nu*delta and theta - nu*delta on the same layout seed."""
    env_id, theta_plus, theta_minus, ep_seed = task_args
    env = _WORKER_ENVS[env_id]
    p_plus = LinearPolicy()
    p_plus.set_flat(theta_plus)
    p_minus = LinearPolicy()
    p_minus.set_flat(theta_minus)

    res_plus = rollout_episode(env, p_plus, seed=ep_seed, cbf=_WORKER_CBF, cbf_penalty=_WORKER_CBF_PENALTY)
    res_minus = rollout_episode(env, p_minus, seed=ep_seed, cbf=_WORKER_CBF, cbf_penalty=_WORKER_CBF_PENALTY)
    return (
        {
            'return': res_plus.episode_return,
            'cost': res_plus.episode_cost,
            'goals': res_plus.goals,
            'fell': float(res_plus.fell),
            'length': res_plus.length,
        },
        {
            'return': res_minus.episode_return,
            'cost': res_minus.episode_cost,
            'goals': res_minus.goals,
            'fell': float(res_minus.fell),
            'length': res_minus.length,
        },
    )


def _eval_q3_pair_worker(task_args: tuple[str, str, np.ndarray, np.ndarray, int]) -> tuple[dict, dict, dict, dict]:
    """Evaluate theta +/- nu*delta at both nominal (7.4 V) and reduced (6.0 V) strength."""
    env_nom_id, env_red_id, theta_plus, theta_minus, ep_seed = task_args
    p_plus = LinearPolicy()
    p_plus.set_flat(theta_plus)
    p_minus = LinearPolicy()
    p_minus.set_flat(theta_minus)

    def _run(env_id: str, pol: LinearPolicy) -> dict:
        res = rollout_episode(
            _WORKER_ENVS[env_id], pol, seed=ep_seed, cbf=_WORKER_CBF, cbf_penalty=_WORKER_CBF_PENALTY
        )
        return {
            'return': res.episode_return,
            'cost': res.episode_cost,
            'goals': res.goals,
            'fell': float(res.fell),
            'length': res.length,
        }

    return (
        _run(env_nom_id, p_plus),
        _run(env_red_id, p_plus),
        _run(env_nom_id, p_minus),
        _run(env_red_id, p_minus),
    )


def _evaluate_across_levels(
    theta: np.ndarray,
    eval_envs: list[str],
    eval_episodes: int,
    use_cbf: bool,
    cbf_lookahead: float,
    cbf_alpha: float,
) -> dict[str, dict[str, float]]:
    import microduck_safety_gym  # noqa: F401, PLC0415
    import safety_gymnasium  # noqa: PLC0415

    policy = LinearPolicy()
    policy.set_flat(theta)
    cbf = CBFSafetyFilter(lookahead=cbf_lookahead, alpha=cbf_alpha) if use_cbf else None
    out: dict[str, dict[str, float]] = {}
    for eid in eval_envs:
        env = safety_gymnasium.make(eid)
        rets, costs, goals, falls = [], [], [], []
        for ep in range(eval_episodes):
            res = rollout_episode(env, policy, seed=10_000 + ep, cbf=cbf)
            rets.append(res.episode_return)
            costs.append(res.episode_cost)
            goals.append(res.goals)
            falls.append(float(res.fell))
        env.close()
        out[eid] = {
            'return_mean': float(np.mean(rets)),
            'cost_mean': float(np.mean(costs)),
            'goals_mean': float(np.mean(goals)),
            'fall_rate': float(np.mean(falls)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mode', choices=('single', 'q3-standard', 'q3-robust'), default='single')
    parser.add_argument('--env', default='SafetyMicroduckGoal1-v0', help='env id for --mode single')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total-steps', type=int, default=2_000_000)
    parser.add_argument('--num-directions', type=int, default=16, help='N perturbation pairs per iter')
    parser.add_argument('--alpha', type=float, default=0.02, help='ARS primal step size')
    parser.add_argument('--nu', type=float, default=0.03, help='ARS exploration noise std')
    parser.add_argument('--eta', type=float, default=0.05, help='dual step size for lambda')
    parser.add_argument('--lambda-init', type=float, default=0.0, help='initial Lagrange multiplier')
    parser.add_argument('--cost-limit', type=float, default=25.0, help='episode cost budget d')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--cbf', action='store_true', help='apply M3 CBF filter during training')
    parser.add_argument('--cbf-lookahead', type=float, default=0.15)
    parser.add_argument('--cbf-alpha', type=float, default=1.0)
    parser.add_argument('--cbf-penalty', type=float, default=0.0, help='beta * ||a_safe - a_des|| penalty')
    parser.add_argument('--eval-every', type=int, default=100_000, help='checkpoint/eval period in env steps')
    parser.add_argument('--eval-episodes', type=int, default=10, help='episodes per level during periodic eval (0 to skip)')
    parser.add_argument('--log-dir', type=Path, default=Path('runs'))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LinearPolicy()
    theta = policy.get_flat()

    nom_env = 'SafetyMicroduckGoal1-v0'
    red_env = 'SafetyMicroduckGoal1MismatchBattery-v0'
    worker_envs = [args.env] if args.mode == 'single' else [nom_env, red_env]

    stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    tag = args.env if args.mode == 'single' else args.mode
    if args.cbf:
        tag += '+CBF'
    run_dir = args.log_dir / f'M2-{tag}' / f'seed-{args.seed:03d}-{stamp}'
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    csv_path = run_dir / 'progress.csv'
    csv_fields = [
        'iteration',
        'env_steps',
        'cumulative_cost',
        'return_mean',
        'cost_mean',
        'cost_nom',
        'cost_red',
        'goals_mean',
        'fall_rate',
        'lambda',
        'lambda_nom',
        'lambda_red',
    ]
    eval_levels = [
        'SafetyMicroduckGoal1-v0',
        'SafetyMicroduckGoal1MismatchModerate-v0',
        'SafetyMicroduckGoal1MismatchSevere-v0',
    ]

    lam = float(args.lambda_init)
    lam_nom = float(args.lambda_init)
    lam_red = float(args.lambda_init)
    total_steps = 0
    cumulative_cost = 0.0
    next_eval_steps = args.eval_every
    iteration = 0

    policy.save(ckpt_dir / 'step-0.npz', env_steps=0, lam=lam, lam_nom=lam_nom, lam_red=lam_red)

    with (
        csv_path.open('w', newline='') as f,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_init_worker,
            initargs=(worker_envs, args.cbf, args.cbf_lookahead, args.cbf_alpha, args.cbf_penalty),
        ) as pool,
    ):
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        while total_steps < args.total_steps:
            iteration += 1
            deltas = rng.standard_normal((args.num_directions, policy.num_params))
            base_seed = args.seed * 1_000_000 + iteration * args.num_directions

            if args.mode == 'single':
                tasks = [
                    (args.env, theta + args.nu * deltas[k], theta - args.nu * deltas[k], base_seed + k)
                    for k in range(args.num_directions)
                ]
                results = list(pool.map(_eval_pair_worker, tasks))
                scores_plus = np.array([r[0]['return'] - lam * r[0]['cost'] for r in results])
                scores_minus = np.array([r[1]['return'] - lam * r[1]['cost'] for r in results])
                all_eps = [ep for pair in results for ep in pair]
                mean_cost = float(np.mean([ep['cost'] for ep in all_eps]))
                cost_nom = mean_cost
                cost_red = float('nan')
                lam = max(0.0, lam + args.eta * (mean_cost - args.cost_limit))
            else:
                tasks = [
                    (nom_env, red_env, theta + args.nu * deltas[k], theta - args.nu * deltas[k], base_seed + k)
                    for k in range(args.num_directions)
                ]
                results = list(pool.map(_eval_q3_pair_worker, tasks))
                if args.mode == 'q3-standard':
                    scores_plus = np.array(
                        [0.5 * (r[0]['return'] + r[1]['return']) - lam * 0.5 * (r[0]['cost'] + r[1]['cost']) for r in results]
                    )
                    scores_minus = np.array(
                        [0.5 * (r[2]['return'] + r[3]['return']) - lam * 0.5 * (r[2]['cost'] + r[3]['cost']) for r in results]
                    )
                else:
                    scores_plus = np.array(
                        [
                            0.5 * (r[0]['return'] + r[1]['return']) - lam_nom * r[0]['cost'] - lam_red * r[1]['cost']
                            for r in results
                        ]
                    )
                    scores_minus = np.array(
                        [
                            0.5 * (r[2]['return'] + r[3]['return']) - lam_nom * r[2]['cost'] - lam_red * r[3]['cost']
                            for r in results
                        ]
                    )
                nom_eps = [r[0] for r in results] + [r[2] for r in results]
                red_eps = [r[1] for r in results] + [r[3] for r in results]
                all_eps = nom_eps + red_eps
                cost_nom = float(np.mean([ep['cost'] for ep in nom_eps]))
                cost_red = float(np.mean([ep['cost'] for ep in red_eps]))
                mean_cost = 0.5 * (cost_nom + cost_red)
                if args.mode == 'q3-standard':
                    lam = max(0.0, lam + args.eta * (mean_cost - args.cost_limit))
                else:
                    lam_nom = max(0.0, lam_nom + args.eta * (cost_nom - args.cost_limit))
                    lam_red = max(0.0, lam_red + args.eta * (cost_red - args.cost_limit))

            sigma_l = float(np.std(np.concatenate([scores_plus, scores_minus])))
            if sigma_l > 1e-8:
                grad = np.sum((scores_plus - scores_minus)[:, None] * deltas, axis=0) / (
                    args.num_directions * sigma_l
                )
                theta = theta + args.alpha * grad

            step_inc = sum(ep['length'] for ep in all_eps)
            cost_inc = sum(ep['cost'] for ep in all_eps)
            total_steps += step_inc
            cumulative_cost += cost_inc

            row = {
                'iteration': iteration,
                'env_steps': total_steps,
                'cumulative_cost': cumulative_cost,
                'return_mean': float(np.mean([ep['return'] for ep in all_eps])),
                'cost_mean': mean_cost,
                'cost_nom': cost_nom,
                'cost_red': cost_red,
                'goals_mean': float(np.mean([ep['goals'] for ep in all_eps])),
                'fall_rate': float(np.mean([ep['fell'] for ep in all_eps])),
                'lambda': lam,
                'lambda_nom': lam_nom,
                'lambda_red': lam_red,
            }
            writer.writerow(row)
            f.flush()

            print(
                f'iter {iteration:4d} | steps {total_steps:8d}/{args.total_steps} | '
                f'ret {row["return_mean"]:6.2f} | cost {mean_cost:5.1f} | '
                f'goals {row["goals_mean"]:4.2f} | lam {lam:5.2f} ({lam_nom:4.2f}/{lam_red:4.2f})'
            )

            if total_steps >= next_eval_steps or total_steps >= args.total_steps:
                policy.set_flat(theta)
                ckpt_path = ckpt_dir / f'step-{total_steps}.npz'
                policy.save(ckpt_path, env_steps=total_steps, lam=lam, lam_nom=lam_nom, lam_red=lam_red)
                if args.eval_episodes > 0:
                    eval_res = _evaluate_across_levels(
                        theta,
                        eval_levels,
                        args.eval_episodes,
                        args.cbf,
                        args.cbf_lookahead,
                        args.cbf_alpha,
                    )
                    (ckpt_dir / f'eval-{total_steps}.json').write_text(json.dumps(eval_res, indent=2))
                next_eval_steps += args.eval_every

    print(f'Finished training. Artifacts written to {run_dir}')


if __name__ == '__main__':
    main()
