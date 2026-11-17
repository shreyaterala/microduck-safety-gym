"""Low-level controller: Pollen's pretrained walking policy driving the BAM actuators.

The policy (``velstand.onnx``) maps a 61-D observation to 14 joint-position offsets at 50 Hz:

    obs = [base_ang_vel(3), projected_gravity(3), joint_pos - default(14), joint_vel(14),
           last_action(14), command(13)]
    command = [twist (vx, vy, yaw_rate)(3), head_pose(4), body_pose(6)]  # head/body left at 0
    joint_target = default_joint_pos + action_scale * action

This mirrors ``microduck_rl/scripts/infer_policy.py`` (``PolicyInference`` with projected
gravity and ``--new-cmd-obs``), which Pollen uses to rehearse deployments on CPU MuJoCo.
The observation normalizer is baked into the ONNX file.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_safety_gym import model as duck_model

COMMAND_DIM = 13


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of ``quat`` (w, x, y, z)."""
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


class WalkingController:
    """Runs the walking policy and the BAM actuator model on a MuJoCo model/data pair."""

    def __init__(self, policy_path: str | Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(policy_path),
            sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = meta['joint_names'].split(',')
        self.default_pose = np.array(
            [float(v) for v in meta['default_joint_pos'].split(',')],
            dtype=np.float32,
        )
        self.action_scale = float(meta.get('action_scale', 1.0))
        obs_dim = self.session.get_inputs()[0].shape[1]
        if obs_dim != 6 + 3 * len(self.joint_names) + COMMAND_DIM:
            raise ValueError(f'{policy_path}: expected the 61-D microduck obs, got {obs_dim}')

        self.bam_model = duck_model.load_bam_model()
        self.bam = None
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.command = np.zeros(COMMAND_DIM, dtype=np.float32)

    def bind(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        vin: float = duck_model.BAM_VIN,
    ) -> None:
        """Attach to a (freshly built) model; Safety Gymnasium recompiles the world every reset.

        ``vin`` is the servo supply voltage; the load-dependent drop keeps the nominal headroom
        (7.4 V -> 6.0 V floor), shifted down with it.
        """
        from bam.mujoco import MujocoController  # noqa: PLC0415

        self.model, self.data = model, data
        self.bam_model.actuator.vin = vin
        self.bam = MujocoController(
            self.bam_model,
            self.joint_names,
            model,
            data,
            vin_drop_gain=duck_model.BAM_VIN_DROP_GAIN,
            vin_min=vin - (duck_model.BAM_VIN - duck_model.BAM_VIN_MIN),
        )
        jids = [model.joint(n).id for n in self.joint_names]
        self.qpos_idx = model.jnt_qposadr[jids]
        self.qvel_idx = model.jnt_dofadr[jids]
        self.trunk_id = model.body('agent').id
        gyro = model.sensor('imu_ang_vel')
        self.gyro_slice = slice(gyro.adr[0], gyro.adr[0] + 3)
        self.last_action[:] = 0.0
        self.command[:] = 0.0
        self.bam.q_target[:] = self.default_pose

    def set_twist(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Set the velocity command (m/s, m/s, rad/s) the policy tracks."""
        self.command[0:3] = (vx, vy, yaw_rate)

    def observation(self) -> np.ndarray:
        """The 61-D policy observation for the current state."""
        data = self.data
        gravity = _quat_rotate_inverse(
            data.xquat[self.trunk_id],
            np.array([0.0, 0.0, -1.0]),
        )
        return np.concatenate(
            [
                data.sensordata[self.gyro_slice],
                gravity,
                data.qpos[self.qpos_idx] - self.default_pose,
                data.qvel[self.qvel_idx],
                self.last_action,
                self.command,
            ],
        ).astype(np.float32)

    def policy_step(self) -> None:
        """One 50 Hz policy tick: new joint targets for the actuators."""
        obs = self.observation()[None, :]
        action = self.session.run([self._output], {self._input: obs})[0][0].astype(np.float32)
        self.last_action = action
        self.bam.q_target[:] = self.default_pose + self.action_scale * action

    def actuator_step(self) -> None:
        """Before every physics step: BAM computes motor torques (and joint friction)."""
        self.bam.update()
