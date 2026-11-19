"""Unit tests for B0 (LidarGoalPolicy), M2 (LinearPolicy), and M3 (CBFSafetyFilter / 2-D QP)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from microduck_safety_gym.controllers import (
    CBFSafetyFilter,
    LidarGoalPolicy,
    LinearPolicy,
    solve_2d_cbf_qp,
    wrap_pi,
)


def test_wrap_pi():
    assert wrap_pi(0.0) == pytest.approx(0.0)
    assert wrap_pi(np.pi) == pytest.approx(np.pi)
    assert wrap_pi(-np.pi) == pytest.approx(np.pi)
    assert wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)


def test_lidar_goal_policy():
    policy = LidarGoalPolicy(goal_slice=slice(0, 16), k_psi=2.0, psi_0=0.6)

    # Strongest bin 0 -> straight ahead -> walk forward, zero turn
    obs = np.zeros(20)
    obs[0] = 0.8
    act = policy(obs)
    assert act[0] == 1.0 and act[1] == pytest.approx(0.0)

    # Strongest bin 4 -> +pi/2 (left) -> stop walking, turn left
    obs[:] = 0.0
    obs[4] = 0.7
    act = policy(obs)
    assert act[0] == 0.0 and act[1] == pytest.approx(1.0)

    # Strongest bin 12 -> -pi/2 (right) -> stop walking, turn right
    obs[:] = 0.0
    obs[12] = 0.7
    act = policy(obs)
    assert act[0] == 0.0 and act[1] == pytest.approx(-1.0)


def test_linear_policy_roundtrip(tmp_path):
    pol = LinearPolicy(obs_dim=60, act_dim=2)
    assert pol.num_params == 122
    theta = np.linspace(-0.5, 0.5, 122)
    pol.set_flat(theta)
    obs = np.ones(60)
    out = pol(obs)
    assert out.shape == (2,)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)

    ckpt = tmp_path / 'pol.npz'
    pol.save(ckpt, env_steps=100)
    loaded = LinearPolicy.load(ckpt)
    assert np.allclose(loaded.get_flat(), theta)
    assert np.allclose(loaded(obs), out)


def test_solve_2d_cbf_qp_feasible_and_box():
    # Unconstrained feasible
    u_des = np.array([0.8, 0.2])
    A = np.array([[1.0, 0.0]])
    b = np.array([0.5])
    assert np.allclose(solve_2d_cbf_qp(u_des, A, b), u_des)

    # Single halfspace: x >= 0.5 from u_des = [0.0, 0.3] -> [0.5, 0.3]
    u = solve_2d_cbf_qp(np.array([0.0, 0.3]), A, b)
    assert np.allclose(u, [0.5, 0.3], atol=1e-5)

    # Two intersecting constraints + box: x <= 0.2 (-x >= -0.2), y >= 0.4
    A2 = np.array([[-1.0, 0.0], [0.0, 1.0]])
    b2 = np.array([-0.2, 0.4])
    u2 = solve_2d_cbf_qp(np.array([0.9, -0.5]), A2, b2)
    assert np.allclose(u2, [0.2, 0.4], atol=1e-5)


def test_solve_2d_cbf_qp_infeasible_recovery():
    # Impossible constraint x >= 2.0 when box is [-1, 1] -> should push x to +1.0
    A = np.array([[1.0, 0.0]])
    b = np.array([2.0])
    u = solve_2d_cbf_qp(np.array([-0.5, 0.2]), A, b)
    assert u[0] == pytest.approx(1.0, abs=1e-3)
    assert -1.0 <= u[1] <= 1.0


def test_cbf_safety_filter_overrides_head_on_hazard():
    cbf = CBFSafetyFilter(lookahead=0.15, alpha=1.0)

    # Robot at origin facing +x, hazard directly ahead at (0.36, 0.0) with radius 0.20 -> q=(0.15, 0)
    # h = (0.15 - 0.36)^2 - 0.35^2 = 0.0441 - 0.1225 < 0? Wait: place hazard at (0.55, 0.0) so h > 0:
    # (0.15 - 0.55)^2 - 0.35^2 = 0.16 - 0.1225 = 0.0375 > 0.
    task = SimpleNamespace(
        agent=SimpleNamespace(pos=np.array([0.0, 0.0, 0.1]), mat=np.eye(3)),
        hazards=SimpleNamespace(pos=np.array([[0.55, 0.0, 0.0]]), size=0.20),
    )
    a_safe, info = cbf.filter(task, np.array([1.0, 0.0]))
    assert info['cbf_overridden'] is True
    assert info['cbf_min_h'] > 0.0
    # Must slow down forward speed to satisfy CBF constraint
    assert a_safe[0] < 1.0
    assert info['incursion_depth'] == 0.0

    # Far away hazard -> no override
    task_far = SimpleNamespace(
        agent=SimpleNamespace(pos=np.array([0.0, 0.0, 0.1]), mat=np.eye(3)),
        hazards=SimpleNamespace(pos=np.array([[2.0, 2.0, 0.0]]), size=0.20),
    )
    a_far, info_far = cbf.filter(task_far, np.array([1.0, 0.0]))
    assert info_far['cbf_overridden'] is False
    assert np.allclose(a_far, [1.0, 0.0])
