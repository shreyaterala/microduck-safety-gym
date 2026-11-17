"""The Microduck Goal1 env behaves like a Safety Gymnasium robot env.

Run with:  python -m pytest tests -q   (about a minute; each reset compiles the world)
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import microduck_safety_gym  # noqa: F401
import safety_gymnasium


@pytest.fixture(scope='module')
def env():
    env = safety_gymnasium.make('SafetyMicroduckGoal1-v0')
    yield env
    env.close()


def test_registered_ids():
    for task in ('GoalBase', 'Goal0', 'Goal1', 'Goal2'):
        assert f'SafetyMicroduck{task}-v0' in microduck_safety_gym.ENV_IDS


def test_spaces_match_point(env):
    point = safety_gymnasium.make('SafetyPointGoal1-v0')
    assert env.observation_space.shape == point.observation_space.shape == (60,)
    assert env.action_space.shape == point.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert env.spec.max_episode_steps == point.spec.max_episode_steps == 1000
    point.close()


def test_control_period_is_20ms(env):
    env.reset(seed=0)
    task = env.unwrapped.task
    t0 = task.data.time
    env.step(np.zeros(2))
    assert task.data.time - t0 == pytest.approx(0.02)


def test_reset_stands_upright(env):
    env.reset(seed=1)
    task = env.unwrapped.task
    assert task.agent.is_alive()
    assert 0.10 < task.data.body('agent').xpos[2] < 0.13


def test_forward_action_walks_forward(env):
    env.reset(seed=2)
    task = env.unwrapped.task
    start = task.agent.pos[:2].copy()
    heading = task.agent.mat[:2, 0].copy()
    for _ in range(200):  # 4 s at full forward command
        _, _, _, terminated, _, _ = env.step(np.array([1.0, 0.0]))
        assert not terminated
    assert np.dot(task.agent.pos[:2] - start, heading) > 0.2


def test_fall_terminates(env):
    env.reset(seed=3)
    task = env.unwrapped.task
    qadr = task.model.joint('trunk_base_freejoint').qposadr[0]
    task.data.qpos[qadr + 3 : qadr + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # 90 deg roll
    mujoco.mj_forward(task.model, task.data)
    _, _, _, terminated, _, _ = env.step(np.zeros(2))
    assert terminated


def test_mismatch_levels():
    env = safety_gymnasium.make('SafetyMicroduckGoal1MismatchSevere-v0')
    agent = env.unwrapped.task.agent
    assert agent.mismatch.command_delay_steps == 5 and agent.mismatch.battery_vin == 6.0
    env.reset(seed=0)
    assert agent.controller.bam_model.actuator.vin == 6.0
    model = env.unwrapped.task.model
    assert model.geom_friction[model.geom('floor').id, 0] == 0.3
    commands = []
    for _ in range(6):
        env.step(np.array([1.0, 0.0]))
        commands.append(agent.command[0])
    assert commands[4] == 0 and commands[5] > 0  # command arrives 5 steps (100 ms) late
    env.close()


def test_hazard_costs(env):
    env.reset(seed=4)
    task = env.unwrapped.task
    qadr = task.model.joint('trunk_base_freejoint').qposadr[0]
    task.data.qpos[qadr : qadr + 2] = task.hazards.pos[0][:2]
    mujoco.mj_forward(task.model, task.data)
    _, _, cost, _, _, info = env.step(np.zeros(2))
    assert cost > 0 and info['cost_hazards'] > 0
