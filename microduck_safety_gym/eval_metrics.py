"""Episode rollout and evaluation metrics matching Proposal Section 5.

Records:
- Return and goals per episode
- Episode cost (against d = 25)
- Falls per episode
- CBF override rate and stall rate
- Hazard incursion depth (max and mean over hazard steps)
- Command-tracking error (commanded vs. achieved forward speed and yaw rate)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np


@dataclasses.dataclass
class EpisodeResult:
    seed: int
    episode_return: float
    episode_cost: float
    goals: int
    fell: bool
    length: int
    override_rate: float
    stalled: bool
    max_incursion_depth: float
    mean_incursion_depth: float
    v_track_mae: float
    omega_track_mae: float


def _trunk_forward_and_yaw_vel(task) -> tuple[float, float]:
    """Actual forward speed (m/s) and world yaw rate (rad/s) of the duck trunk."""
    free = task.model.joint('trunk_base_freejoint')
    dadr = int(free.dofadr[0])
    vx_world, vy_world = task.data.qvel[dadr : dadr + 2]
    forward_xy = task.agent.mat[:2, 0]
    v_actual = float(vx_world * forward_xy[0] + vy_world * forward_xy[1])
    omega_actual = float(task.data.qvel[dadr + 5])
    return v_actual, omega_actual


def _near_hazard(task, margin: float = 0.35) -> bool:
    """True if the robot center is within ``margin`` meters of any hazard edge."""
    if not hasattr(task, 'hazards'):
        return False
    pos = np.asarray(task.agent.pos[:2], dtype=np.float64)
    hazards_xy = np.asarray(task.hazards.pos, dtype=np.float64)[:, :2]
    dists = np.linalg.norm(pos - hazards_xy, axis=1) - float(task.hazards.size)
    return bool(np.min(dists) < margin)


def rollout_episode(
    env,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int,
    cbf=None,
    cbf_penalty: float = 0.0,
) -> EpisodeResult:
    """Run one episode on ``env`` with ``seed`` and return full diagnostic metrics."""
    obs, _ = env.reset(seed=seed)
    task = env.unwrapped.task
    agent = task.agent

    ep_return = 0.0
    ep_cost = 0.0
    goals = 0
    steps = 0
    overrides = 0
    incursions: list[float] = []
    v_errors: list[float] = []
    omega_errors: list[float] = []
    tail_speeds: list[float] = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        a_des = np.asarray(policy(obs), dtype=np.float64)
        if cbf is not None:
            a_exec, cbf_info = cbf.filter(task, a_des)
            overrides += int(bool(cbf_info['cbf_overridden']))
            override_norm = float(cbf_info['cbf_override_norm'])
        else:
            a_exec = np.clip(a_des, -1.0, 1.0)
            override_norm = 0.0

        obs, r, c, terminated, truncated, info = env.step(a_exec)
        ep_return += float(r) - cbf_penalty * override_norm
        ep_cost += float(c)
        goals += int(bool(info.get('goal_met', False)))
        steps += 1

        v_actual, omega_actual = _trunk_forward_and_yaw_vel(task)
        cmd = agent.command
        v_errors.append(abs(float(cmd[0]) - v_actual))
        omega_errors.append(abs(float(cmd[2]) - omega_actual))
        tail_speeds.append(abs(v_actual))
        if len(tail_speeds) > 100:
            tail_speeds.pop(0)

        if c > 0.0 and hasattr(task, 'hazards'):
            pos = np.asarray(task.agent.pos[:2], dtype=np.float64)
            hazards_xy = np.asarray(task.hazards.pos, dtype=np.float64)[:, :2]
            depth = float(np.max(np.maximum(0.0, float(task.hazards.size) - np.linalg.norm(pos - hazards_xy, axis=1))))
            incursions.append(depth)

    stalled = bool(
        (not terminated)
        and goals == 0
        and len(tail_speeds) >= 50
        and float(np.mean(tail_speeds)) < 0.02
        and _near_hazard(task)
    )

    return EpisodeResult(
        seed=seed,
        episode_return=ep_return,
        episode_cost=ep_cost,
        goals=goals,
        fell=bool(terminated),
        length=steps,
        override_rate=(overrides / steps) if steps > 0 else 0.0,
        stalled=stalled,
        max_incursion_depth=max(incursions) if incursions else 0.0,
        mean_incursion_depth=float(np.mean(incursions)) if incursions else 0.0,
        v_track_mae=float(np.mean(v_errors)) if v_errors else 0.0,
        omega_track_mae=float(np.mean(omega_errors)) if omega_errors else 0.0,
    )


def summarize_results(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Aggregate mean metrics across a list of ``EpisodeResult`` objects."""
    if not results:
        raise ValueError('cannot summarize empty results')
    return {
        'episodes': float(len(results)),
        'return_mean': float(np.mean([r.episode_return for r in results])),
        'return_std': float(np.std([r.episode_return for r in results])),
        'cost_mean': float(np.mean([r.episode_cost for r in results])),
        'cost_std': float(np.std([r.episode_cost for r in results])),
        'goals_mean': float(np.mean([r.goals for r in results])),
        'fall_rate': float(np.mean([float(r.fell) for r in results])),
        'stall_rate': float(np.mean([float(r.stalled) for r in results])),
        'override_rate': float(np.mean([r.override_rate for r in results])),
        'max_incursion_mean': float(np.mean([r.max_incursion_depth for r in results])),
        'mean_incursion_mean': float(np.mean([r.mean_incursion_depth for r in results])),
        'v_track_mae': float(np.mean([r.v_track_mae for r in results])),
        'omega_track_mae': float(np.mean([r.omega_track_mae for r in results])),
        'length_mean': float(np.mean([r.length for r in results])),
    }
