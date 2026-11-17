"""The Microduck as a Safety Gymnasium agent.

Interface, kept as close to Point/Car as the robot allows:

- **Action:** ``Box(-1, 1, (2,))`` = (forward, turn), like Point's two actuators. It is scaled
  to a velocity command for the pretrained walking policy: forward speed ``a[0] * 0.4`` m/s and
  yaw rate ``a[1] * 1.0`` rad/s (velstand's training ranges). Lateral velocity is held at 0.
- **Observation:** the same four agent sensors as Point and Car (accelerometer, velocimeter,
  gyro, magnetometer on the ``agent`` site, which sits on the trunk) plus the task's lidars.
  For Goal1 that gives the same 60-D vector as SafetyPointGoal1.
- **Timing:** one env step = one 50 Hz policy tick = 4 x 5 ms physics steps = 20 ms, the same
  control period as Point (10 x 2 ms). Episodes keep Safety Gymnasium's 1,000 steps (20 s).
- **Termination:** like Ant, the agent "dies" (episode terminates) when it falls: trunk below
  6 cm or tilted more than 60 degrees.

Two embodiment details have no Point equivalent:

- The command is slew-rate limited (1 m/s^2, 4 rad/s^2) before it reaches the walking policy.
  The policy was trained on piecewise-constant commands; a command that jumps every 20 ms
  (e.g. an exploring RL policy) is far outside that distribution and trips the gait.
- On reset the duck is placed in its standing pose and the walking policy runs for 0.5 s at
  zero command, so every episode starts from a settled stance rather than a drop.

``mismatch`` (see ``mismatch.py``) optionally adds command delay, battery sag and random shoves.
It defaults to none; the ``...Mismatch{Moderate,Severe}-v0`` env ids set it.
"""

from __future__ import annotations

import collections

import glfw
import mujoco
import numpy as np
from gymnasium import spaces

from microduck_safety_gym import model as duck_model
from microduck_safety_gym import paths
from microduck_safety_gym._compat import import_safety_gymnasium
from microduck_safety_gym.controller import WalkingController
from microduck_safety_gym.mismatch import Mismatch

import_safety_gymnasium()
from safety_gymnasium.bases.base_agent import BaseAgent  # noqa: E402
from safety_gymnasium.utils.random_generator import RandomGenerator  # noqa: E402


class Microduck(BaseAgent):
    """Pollen Robotics' Microduck biped, steered through its pretrained walking policy."""

    # Command scaling: action in [-1, 1] -> velstand's training command ranges.
    max_forward_speed = 0.4  # m/s (trained range is +-0.4)
    max_yaw_rate = 1.0  # rad/s (trained range is +-1.0)
    # Slew-rate limits on the command seen by the walking policy.
    max_linear_accel = 1.0  # m/s^2
    max_yaw_accel = 4.0  # rad/s^2
    # Fall detection.
    min_trunk_height = 0.06  # m (standing trunk height is ~0.116)
    min_upright_cos = 0.5  # cos(60 deg)
    settle_ticks = 25  # 0.5 s of standing before the episode starts

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list | None = None,
        locations: list | None = None,
        keepout: float = 0.4,
        rot: float | None = None,
    ) -> None:
        self._xml_path = duck_model.build_agent_xml()
        self.controller = WalkingController(paths.policy_path())
        self._target = np.zeros(3)
        self._command = np.zeros(3)
        self.mismatch = Mismatch()
        self._delayed: collections.deque = collections.deque()
        self._next_shove = np.inf
        self._shove_end = -np.inf
        super().__init__(
            self.__class__.__name__,
            random_generator,
            placements,
            locations,
            keepout,
            rot,
        )
        # Action noise (Underlying.action_noise) is drawn with size body_info.nu.
        self.body_info.nu = self.action_space.shape[0]

    # ----------------------------------------------------------------- model / spaces
    def _load_model(self) -> None:
        # BaseAgent joins BASE_DIR with self.base; an absolute path survives os.path.join.
        self.base = str(self._xml_path)
        super()._load_model()

    def _build_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    # ----------------------------------------------------------------- control
    def apply_action(self, action: np.ndarray, noise: np.ndarray | None = None) -> None:
        """Set the target velocity command from a (forward, turn) action in [-1, 1]."""
        action = np.asarray(action, dtype=np.float64)
        if noise is not None:
            action = action + noise
        action = np.clip(action, -1.0, 1.0)
        target = np.array([action[0] * self.max_forward_speed, 0.0, action[1] * self.max_yaw_rate])
        if self.mismatch.command_delay_steps > 0:
            # The queue holds the last `delay` targets; the oldest one reaches the walking policy.
            self._delayed.append(target)
            target = self._delayed.popleft()
        self._target[:] = target

    def control_tick(self) -> None:
        """Once per env step: slew the command toward the target and run the walking policy."""
        dt = duck_model.PHYSICS_DT * duck_model.DECIMATION
        limit = np.array([self.max_linear_accel, self.max_linear_accel, self.max_yaw_accel]) * dt
        self._command += np.clip(self._target - self._command, -limit, limit)
        self.controller.set_twist(*self._command)
        self.controller.policy_step()

    def physics_step(self) -> None:
        """Before every MuJoCo step: shoves (if enabled), then BAM motor torques."""
        if self.mismatch.shoves:
            self._shove()
        self.controller.actuator_step()

    def _shove(self) -> None:
        """Start, hold or end a horizontal push on the trunk."""
        data = self.engine.data
        trunk = self.controller.trunk_id
        if data.time >= self._shove_end:
            data.xfrc_applied[trunk, :] = 0.0
        if data.time >= self._next_shove:
            angle = self.random_generator.uniform(0.0, 2 * np.pi)
            force = self.mismatch.shove_force
            data.xfrc_applied[trunk, :3] = (force * np.cos(angle), force * np.sin(angle), 0.0)
            self._shove_end = data.time + self.mismatch.shove_duration
            self._next_shove = data.time + self.random_generator.uniform(*self.mismatch.shove_interval)

    @property
    def command(self) -> np.ndarray:
        """The (vx, vy, yaw_rate) command currently fed to the walking policy."""
        return self._command.copy()

    # ----------------------------------------------------------------- episode
    def is_alive(self) -> bool:
        """Alive while the trunk is high enough and within 60 degrees of upright."""
        data = self.engine.data
        trunk = data.body('agent')
        upright = trunk.xmat.reshape(3, 3)[2, 2]
        return bool(trunk.xpos[2] > self.min_trunk_height and upright > self.min_upright_cos)

    def reset(self) -> None:
        """Stand the duck up where Safety Gymnasium placed it, then let it settle."""
        model, data = self.engine.model, self.engine.data
        # The floor Pollen trains on is a default plane (condim 3); Safety Gymnasium uses 6.
        model.geom_condim[model.geom('floor').id] = 3
        if self.mismatch.floor_friction is not None:
            # MuJoCo uses the larger friction of a contact pair, so lower both the floor and
            # every duck geom. Other objects (vases) keep theirs.
            duck_root = model.body('agent').rootid[0]
            duck_geoms = model.body_rootid[model.geom_bodyid] == duck_root
            model.geom_friction[duck_geoms, 0] = self.mismatch.floor_friction
            model.geom_friction[model.geom('floor').id, 0] = self.mismatch.floor_friction

        # Bind first: BAM's MujocoController calls mj_setConst, which leaves data.qpos at qpos0
        # and would undo the standing pose set below.
        self.controller.bind(model, data, vin=self.mismatch.battery_vin)

        free = model.joint('trunk_base_freejoint')
        qadr = free.qposadr[0]
        data.qpos[qadr + 2] = duck_model.SPAWN_HEIGHT  # keep Safety Gymnasium's x, y and yaw
        jids = [model.joint(n).id for n in self.controller.joint_names]
        data.qpos[model.jnt_qposadr[jids]] = self.controller.default_pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        self._target[:] = 0.0
        self._command[:] = 0.0
        self._next_shove = np.inf  # no shoves while settling
        self._shove_end = -np.inf
        for _ in range(self.settle_ticks):
            self.control_tick()
            for _ in range(duck_model.DECIMATION):
                self.physics_step()
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        self._delayed = collections.deque(
            np.zeros(3) for _ in range(self.mismatch.command_delay_steps)
        )
        if self.mismatch.shoves:
            self._next_shove = data.time + self.random_generator.uniform(*self.mismatch.shove_interval)

    def debug(self) -> None:
        """Keyboard control in the Debug env variants (I/K forward/back, J/L turn)."""
        action = np.zeros(2)
        for key in self.debug_info.keys:
            if key == glfw.KEY_I:
                action[0] += 1
            elif key == glfw.KEY_K:
                action[0] -= 1
            elif key == glfw.KEY_J:
                action[1] += 1
            elif key == glfw.KEY_L:
                action[1] -= 1
        self.apply_action(action)
