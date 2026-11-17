"""Pollen Robotics' Microduck as a Safety Gymnasium agent.

Importing this package (before ``safety_gymnasium`` or ``omnisafe``):

1. makes safety-gymnasium 1.0.0 importable on Python 3.12 (see ``_compat``);
2. adds ``Microduck`` to ``safety_gymnasium.agents`` next to Point, Car, Doggo, Racecar, Ant;
3. registers ``SafetyMicroduck{GoalBase,Goal0,Goal1,Goal2}-v0`` (plus the ``Gymnasium`` and
   ``Vision`` variants), using the same task configs and 1,000-step limit as the other robots;
4. registers ``SafetyMicroduck<task>Mismatch<Level>-v0``: the same tasks with command delay,
   battery sag, shoves and low friction (``mismatch.py``). Levels: ``Moderate`` and ``Severe``
   (all knobs), ``Delay``/``Battery``/``Shoves``/``Friction`` (one knob at its severe value),
   and ``BatteryMix`` (each episode at 7.4 V or 6.0 V, 50/50).

    import microduck_safety_gym  # noqa: F401
    import safety_gymnasium
    env = safety_gymnasium.make('SafetyMicroduckGoal1-v0')
"""

from __future__ import annotations

import copy

from microduck_safety_gym._compat import import_safety_gymnasium
from microduck_safety_gym.agent import Microduck
from microduck_safety_gym.model import DECIMATION

safety_gymnasium = import_safety_gymnasium()
from safety_gymnasium import agents as _sg_agents  # noqa: E402
from safety_gymnasium.utils.registration import safe_registry  # noqa: E402

__all__ = ['ENV_IDS', 'Microduck']

ROBOT_NAME = 'Microduck'
_ENTRY_POINT = 'microduck_safety_gym.builder:MicroduckBuilder'
# Same task table as safety_gymnasium/__init__.py's goal_tasks.
_GOAL_TASKS = {'GoalBase': {'task_name': 'GoalBase'}, 'Goal0': {}, 'Goal1': {}, 'Goal2': {}}
_AGENT_CONFIG = {
    'agent_name': ROBOT_NAME,
    # 4 x 5 ms physics steps per env step (one 50 Hz walking-policy tick), deterministic like
    # the other robots' default frameskip_binom_p = 1.0.
    'sim_conf.frameskip_binom_n': DECIMATION,
    'sim_conf.frameskip_binom_p': 1.0,
}

ENV_IDS: list[str] = []
# Env-id suffixes (Mismatch<Level>) for the levels in mismatch.LEVELS: the two combined levels,
# each knob alone at its severe value, and the Q3 50/50 battery training mix.
MISMATCH_LEVELS = ('Moderate', 'Severe', 'Delay', 'Battery', 'Shoves', 'Friction', 'BatteryMix')


def _register(env_id: str, config: dict, task_id: str) -> None:
    """Mirror of safety_gymnasium's ``__register_helper`` with the Microduck entry point."""
    if env_id in safe_registry:
        return
    name, dash, version = env_id.partition('-')
    spec_kwargs = {'config': config, 'task_id': task_id}
    safety_gymnasium.register(
        id=env_id,
        entry_point=_ENTRY_POINT,
        kwargs=spec_kwargs,
        max_episode_steps=1000,
    )
    safety_gymnasium.gymnasium_register(
        id=f'{name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{name}Gymnasium{dash}{version}', **copy.deepcopy(spec_kwargs)},
        max_episode_steps=1000,
    )
    ENV_IDS.append(env_id)


def _register_all() -> None:
    _sg_agents.Microduck = Microduck
    for task_name, task_config in _GOAL_TASKS.items():
        env_id = f'Safety{ROBOT_NAME}{task_name}-v0'
        config = {**copy.deepcopy(task_config), **_AGENT_CONFIG}
        _register(env_id, config, task_id=env_id)
        # As upstream: the Vision variant keeps the base env id as task_id (it names the task class).
        vision = {'observe_vision': True, 'observation_flatten': False, **config}
        _register(f'Safety{ROBOT_NAME}{task_name}Vision-v0', vision, task_id=env_id)
        for level in MISMATCH_LEVELS:
            _register(
                f'Safety{ROBOT_NAME}{task_name}Mismatch{level}-v0',
                {**config, 'mismatch': level.lower()},
                task_id=env_id,
            )


_register_all()
