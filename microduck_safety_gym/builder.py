"""Safety Gymnasium Builder whose physics loop also runs the Microduck's low-level controller.

Safety Gymnasium steps physics in ``Underlying.simulation_forward``: ``apply_action`` once,
then ``frameskip`` calls to ``mj_step``. The duck needs two extra hooks in that loop: one
walking-policy tick per env step, and a BAM torque update before every ``mj_step``. This
Builder installs a copy of ``simulation_forward`` with those hooks. Rewards, costs, goal
resampling, termination, observations and rendering are untouched Safety Gymnasium code.
"""

from __future__ import annotations

import types

import mujoco

from microduck_safety_gym._compat import import_safety_gymnasium
from microduck_safety_gym.mismatch import LEVELS

import_safety_gymnasium()
from safety_gymnasium.builder import Builder  # noqa: E402
from safety_gymnasium.utils.common_utils import MujocoException  # noqa: E402


def _microduck_simulation_forward(task, action):
    """``Underlying.simulation_forward`` with the walking-policy and BAM hooks added."""
    agent = task.agent
    if task.debug:
        agent.debug()
    else:
        noise = (
            task.action_noise * task.random_generator.randn(agent.body_info.nu)
            if task.action_noise
            else None
        )
        agent.apply_action(action, noise)

    agent.control_tick()  # 50 Hz walking-policy tick

    exception = False
    for _ in range(
        task.random_generator.binomial(
            task.sim_conf.frameskip_binom_n,
            task.sim_conf.frameskip_binom_p,
        ),
    ):
        try:
            for mocap in task._mocaps.values():  # noqa: SLF001
                mocap.move()
            agent.physics_step()  # BAM motor torques
            mujoco.mj_step(task.model, task.data)
        except MujocoException as me:
            print('MujocoException', me)
            exception = True
            break
    if exception:
        return exception

    mujoco.mj_forward(task.model, task.data)
    return exception


class MicroduckBuilder(Builder):
    """``safety_gymnasium.builder.Builder`` for the Microduck agent.

    Accepts one extra config key, ``mismatch`` (a key of ``mismatch.LEVELS``: ``'none'``,
    ``'moderate'``, ``'severe'``, a single knob, or ``'batterymix'``), which is taken out before
    Safety Gymnasium parses the rest.
    """

    def __init__(self, task_id: str, config: dict | None = None, **kwargs) -> None:
        config = dict(config or {})
        self.mismatch = LEVELS[config.pop('mismatch', 'none')]
        super().__init__(task_id, config, **kwargs)

    def _get_task(self):
        task = super()._get_task()
        task.simulation_forward = types.MethodType(_microduck_simulation_forward, task)
        task.agent.mismatch = self.mismatch
        return task
