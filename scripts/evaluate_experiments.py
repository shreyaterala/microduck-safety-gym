"""Evaluate B0, B0+M3 (CBF), and M2 checkpoints on fixed layouts across mismatch levels.

Implements the evaluation protocol from Proposal Sections 4 & 5:
- 100 fixed layouts (seeds 0..99) by default.
- Evaluates across combined levels (none, moderate, severe) and single-knob ablations
  (delay, battery, shoves, friction).
- Reports return, goals, cost (vs. d=25), falls, stalls, CBF override rate, incursion depth,
  and command-tracking error (v_track_mae, omega_track_mae).

Examples:
    python scripts/evaluate_experiments.py --method b0 --episodes 100
    python scripts/evaluate_experiments.py --method b0+cbf --episodes 100
    python scripts/evaluate_experiments.py --method m2 --checkpoint runs/M2-.../checkpoints/step-2000000.npz
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path

from microduck_safety_gym.controllers import CBFSafetyFilter, LidarGoalPolicy, LinearPolicy
from microduck_safety_gym.eval_metrics import EpisodeResult, rollout_episode, summarize_results

ALL_LEVEL_ENVS = [
    ('none', 'SafetyMicroduckGoal1-v0'),
    ('moderate', 'SafetyMicroduckGoal1MismatchModerate-v0'),
    ('severe', 'SafetyMicroduckGoal1MismatchSevere-v0'),
    ('delay', 'SafetyMicroduckGoal1MismatchDelay-v0'),
    ('battery', 'SafetyMicroduckGoal1MismatchBattery-v0'),
    ('shoves', 'SafetyMicroduckGoal1MismatchShoves-v0'),
    ('friction', 'SafetyMicroduckGoal1MismatchFriction-v0'),
]


def _run_batch(task_args: tuple) -> list[EpisodeResult]:
    env_id, method, checkpoint, seeds, k_psi, psi_0, lookahead, alpha = task_args
    import microduck_safety_gym  # noqa: F401, PLC0415
    import safety_gymnasium  # noqa: PLC0415

    env = safety_gymnasium.make(env_id)
    if method in ('b0', 'b0+cbf'):
        policy = LidarGoalPolicy.from_env(env, k_psi=k_psi, psi_0=psi_0)
    else:
        policy = LinearPolicy.load(checkpoint)

    cbf = CBFSafetyFilter(lookahead=lookahead, alpha=alpha) if method in ('b0+cbf', 'm2+cbf') else None
    results = [rollout_episode(env, policy, seed=s, cbf=cbf) for s in seeds]
    env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--method', choices=('b0', 'b0+cbf', 'm2', 'm2+cbf'), required=True)
    parser.add_argument('--checkpoint', type=Path, default=None, help='required for m2 / m2+cbf')
    parser.add_argument('--episodes', type=int, default=100, help='number of fixed layouts (seeds 0..N-1)')
    parser.add_argument('--levels', nargs='+', default=[name for name, _ in ALL_LEVEL_ENVS])
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--k-psi', type=float, default=2.0)
    parser.add_argument('--psi-0', type=float, default=0.6)
    parser.add_argument('--cbf-lookahead', type=float, default=0.15)
    parser.add_argument('--cbf-alpha', type=float, default=1.0)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/eval'))
    args = parser.parse_args()

    if args.method in ('m2', 'm2+cbf') and args.checkpoint is None:
        parser.error('--checkpoint is required for m2 / m2+cbf')

    env_map = dict(ALL_LEVEL_ENVS)
    selected = [(lvl, env_map[lvl]) for lvl in args.levels]
    seeds = list(range(args.episodes))
    workers = max(1, args.workers)
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.method.replace('+', '_')
    if args.checkpoint is not None:
        tag += f'_{args.checkpoint.stem}'
    summary_rows: list[dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for level_name, env_id in selected:
            tasks = [
                (
                    env_id,
                    args.method,
                    args.checkpoint,
                    chunk,
                    args.k_psi,
                    args.psi_0,
                    args.cbf_lookahead,
                    args.cbf_alpha,
                )
                for chunk in chunks
            ]
            all_results: list[EpisodeResult] = []
            for batch in pool.map(_run_batch, tasks):
                all_results.extend(batch)
            all_results.sort(key=lambda r: r.seed)

            summary = {'method': args.method, 'level': level_name, 'env_id': env_id, **summarize_results(all_results)}
            summary_rows.append(summary)
            print(
                f'{args.method:7s} | {level_name:8s} | ret {summary["return_mean"]:6.2f} | '
                f'cost {summary["cost_mean"]:5.1f} | goals {summary["goals_mean"]:4.2f} | '
                f'fall {summary["fall_rate"]:4.2f} | stall {summary["stall_rate"]:4.2f} | '
                f'ovr {summary["override_rate"]:4.2f} | inc {summary["max_incursion_mean"]:5.3f}'
            )

    csv_path = args.out_dir / f'{tag}_summary.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = args.out_dir / f'{tag}_summary.json'
    json_path.write_text(json.dumps(summary_rows, indent=2))
    print(f'Wrote {csv_path} and {json_path}')


if __name__ == '__main__':
    main()
