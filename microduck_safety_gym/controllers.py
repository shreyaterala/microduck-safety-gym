"""Controllers and safety filters for Microduck Safety Gymnasium.

Implements the three navigation/safety layers from the project formulation:
- ``LidarGoalPolicy`` (B0): scripted baseline using the 16-bin goal lidar.
- ``LinearPolicy`` (M2): 122-parameter deterministic linear policy for primal-dual random search.
- ``CBFSafetyFilter`` (M3): unicycle look-ahead Control Barrier Function QP filter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def wrap_pi(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if wrapped == -np.pi else wrapped)


def goal_lidar_slice(obs_space_dict, key: str = 'goal_lidar') -> slice:
    """Return the slice of ``key`` inside a flattened Safety Gymnasium observation."""
    offset = 0
    for name, space in obs_space_dict.spaces.items():
        dim = int(np.prod(space.shape))
        if name == key:
            return slice(offset, offset + dim)
        offset += dim
    raise KeyError(f'{key!r} not found in observation space keys: {list(obs_space_dict.spaces)}')


class LidarGoalPolicy:
    """B0 scripted baseline: turn toward the strongest goal-lidar bin and walk.

    Commands:
        omega_t = clip(k_psi * psi_t, -1, 1)
        a_0,t   = 1.0 if |psi_t| < psi_0 else 0.0   (maps to v_t = 0.4 * 1[|psi_t| < psi_0])
    """

    def __init__(
        self,
        goal_slice: slice,
        k_psi: float = 2.0,
        psi_0: float = 0.6,
    ) -> None:
        self.goal_slice = goal_slice
        self.k_psi = k_psi
        self.psi_0 = psi_0

    @classmethod
    def from_env(cls, env, k_psi: float = 2.0, psi_0: float = 0.6) -> LidarGoalPolicy:
        """Construct from a Safety Gymnasium env using its flattened obs_space_dict."""
        obs_dict = env.unwrapped.task.obs_info.obs_space_dict
        return cls(goal_lidar_slice(obs_dict, 'goal_lidar'), k_psi=k_psi, psi_0=psi_0)

    def bearing(self, obs: np.ndarray) -> float | None:
        """Bearing psi_t in (-pi, pi] of the strongest goal-lidar bin, or None if empty."""
        lidar = obs[self.goal_slice]
        if np.max(lidar) <= 0.0:
            return None
        idx = int(np.argmax(lidar))
        bin_size = (2.0 * np.pi) / len(lidar)
        return wrap_pi(idx * bin_size)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        psi = self.bearing(obs)
        if psi is None:
            return np.array([0.0, 1.0], dtype=np.float64)
        omega = float(np.clip(self.k_psi * psi, -1.0, 1.0))
        forward = 1.0 if abs(psi) < self.psi_0 else 0.0
        return np.array([forward, omega], dtype=np.float64)


class LinearPolicy:
    """Deterministic linear policy pi_theta(s) = clip(W s + b, -1, 1) for M2."""

    def __init__(self, obs_dim: int = 60, act_dim: int = 2) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.w_size = obs_dim * act_dim
        self.num_params = self.w_size + act_dim
        self.W = np.zeros((act_dim, obs_dim), dtype=np.float64)
        self.b = np.zeros(act_dim, dtype=np.float64)

    def get_flat(self) -> np.ndarray:
        """Return flattened parameter vector theta = [vec(W), b]."""
        return np.concatenate([self.W.ravel(), self.b])

    def set_flat(self, theta: np.ndarray) -> None:
        """Set W and b from a flattened parameter vector theta."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.num_params,):
            raise ValueError(f'expected shape ({self.num_params},), got {theta.shape}')
        self.W = theta[: self.w_size].reshape(self.act_dim, self.obs_dim).copy()
        self.b = theta[self.w_size :].copy()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(self.W @ np.asarray(obs, dtype=np.float64) + self.b, -1.0, 1.0)

    def save(self, path: str | Path, **metadata) -> None:
        """Save policy weights and optional metadata to an .npz archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.get_flat(), obs_dim=self.obs_dim, act_dim=self.act_dim, **metadata)

    @classmethod
    def load(cls, path: str | Path) -> LinearPolicy:
        """Load policy weights from an .npz archive."""
        data = np.load(path, allow_pickle=False)
        policy = cls(obs_dim=int(data['obs_dim']), act_dim=int(data['act_dim']))
        policy.set_flat(data['theta'])
        return policy


def solve_2d_cbf_qp(
    u_des: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-7,
) -> np.ndarray:
    """Solve min_{u in [low, high]^2} 0.5 ||u - u_des||^2  s.t.  A u >= b.

    Uses Dykstra's cyclic projection onto the box and linear half-spaces when feasible,
    and falls back to minimizing squared constraint violation (with tiny regularization
    toward u_des) if the constrained set is empty (e.g., after an external incursion).
    """
    u_des = np.asarray(u_des, dtype=np.float64)
    u0 = np.clip(u_des, low, high)
    if A.shape[0] == 0 or np.all(A @ u0 >= b - 1e-9):
        return u0

    m = A.shape[0]
    row_norms_sq = np.sum(A * A, axis=1)

    # Dykstra's projection onto {box, halfspace_0, ..., halfspace_{m-1}}
    x = u_des.copy()
    p_box = np.zeros(2, dtype=np.float64)
    p_half = np.zeros((m, 2), dtype=np.float64)

    for _ in range(max_iters):
        x_prev = x.copy()
        # Project onto box [low, high]^2
        y = np.clip(x + p_box, low, high)
        p_box = x + p_box - y
        x = y
        # Project onto each half-space A[i] @ u >= b[i]
        for i in range(m):
            nsq = row_norms_sq[i]
            if nsq <= 1e-12:
                continue
            z = x + p_half[i]
            viol = b[i] - float(A[i] @ z)
            if viol > 0.0:
                y = z + (viol / nsq) * A[i]
            else:
                y = z
            p_half[i] = z - y
            x = y
        if np.linalg.norm(x - x_prev) < tol and np.all(A @ x >= b - 1e-6):
            return np.clip(x, low, high)

    if np.all(A @ x >= b - 1e-5):
        return np.clip(x, low, high)

    # Infeasible fallback (e.g. h_j < 0 due to disturbance): minimize squared violation
    # 0.5 * sum_i max(0, b_i - A_i u)^2 + 1e-4 * 0.5 * ||u - u_des||^2 over [low, high]^2.
    lipschitz = float(np.sum(row_norms_sq)) + 1e-4
    step = 1.0 / max(lipschitz, 1e-6)
    u = np.clip(x, low, high)
    for _ in range(250):
        margins = b - A @ u
        active = np.maximum(0.0, margins)
        grad = -(A.T @ active) + 1e-4 * (u - u_des)
        u_next = np.clip(u - step * grad, low, high)
        if np.linalg.norm(u_next - u) < tol:
            return u_next
        u = u_next
    return u


class CBFSafetyFilter:
    """M3 Control Barrier Function safety filter on a unicycle look-ahead point.

    Maps the robot pose (p, theta) to q = p + l * [cos(theta), sin(theta)]^T,
    defines h_j(q) = ||q - p_j||^2 - (r_j + l)^2 for each hazard j, and minimally
    modifies proposed action a_des in [-1, 1]^2 subject to:
        grad h_j(q)^T q_dot(a) >= -alpha * h_j(q)   for all j.
    """

    def __init__(
        self,
        lookahead: float = 0.15,
        alpha: float = 1.0,
        max_forward_speed: float = 0.4,
        max_yaw_rate: float = 1.0,
    ) -> None:
        self.lookahead = lookahead
        self.alpha = alpha
        self.max_forward_speed = max_forward_speed
        self.max_yaw_rate = max_yaw_rate

    def filter(self, task, a_des: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool]]:
        """Filter proposed action ``a_des`` and return ``(a_safe, info)``."""
        a_des = np.clip(np.asarray(a_des, dtype=np.float64), -1.0, 1.0)
        if not hasattr(task, 'hazards'):
            return a_des, {
                'cbf_overridden': False,
                'cbf_override_norm': 0.0,
                'cbf_min_h': float('inf'),
                'incursion_depth': 0.0,
            }

        pos = np.asarray(task.agent.pos[:2], dtype=np.float64)
        mat = np.asarray(task.agent.mat, dtype=np.float64)
        theta = float(np.arctan2(mat[1, 0], mat[0, 0]))
        c, s = np.cos(theta), np.sin(theta)
        ell = self.lookahead

        # Look-ahead point q and its Jacobian w.r.t. normalized action (a_v, a_omega).
        q = pos + ell * np.array([c, s])
        G_u = np.array(
            [
                [c * self.max_forward_speed, -ell * s * self.max_yaw_rate],
                [s * self.max_forward_speed, ell * c * self.max_yaw_rate],
            ],
            dtype=np.float64,
        )

        hazards_xy = np.asarray(task.hazards.pos, dtype=np.float64)[:, :2]
        r_h = float(task.hazards.size)
        safe_r = r_h + ell

        diff_q = q - hazards_xy  # (N, 2)
        h = np.sum(diff_q * diff_q, axis=1) - safe_r * safe_r
        A = 2.0 * (diff_q @ G_u)  # (N, 2)
        b = -self.alpha * h

        a_safe = solve_2d_cbf_qp(a_des, A, b)
        override_norm = float(np.linalg.norm(a_safe - a_des))

        dist_p = np.linalg.norm(pos - hazards_xy, axis=1)
        incursion = float(np.max(np.maximum(0.0, r_h - dist_p)))

        return a_safe, {
            'cbf_overridden': override_norm > 1e-6,
            'cbf_override_norm': override_norm,
            'cbf_min_h': float(np.min(h)),
            'incursion_depth': incursion,
        }
