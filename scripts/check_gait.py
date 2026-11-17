"""Sanity-check the walking policy outside Safety Gymnasium.

Runs velstand in Pollen's own all-collisions scene (plane floor, BAM actuators, 50 Hz) at a few
constant commands and prints the speed it actually achieves. Use it to separate "the policy
walks this fast" from "the Safety Gymnasium integration slows it down".

    python scripts/check_gait.py
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from microduck_safety_gym import model as duck_model
from microduck_safety_gym import paths
from microduck_safety_gym.controller import WalkingController


def make_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(duck_model.build_agent_xml()))
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    model = spec.compile()
    return model, mujoco.MjData(model)


def run(controller: WalkingController, command: tuple[float, float, float], seconds: float):
    model, data = make_scene()
    # Bind first: BAM's controller calls mj_setConst, which leaves data.qpos at qpos0 and
    # would undo the standing pose.
    controller.bind(model, data)
    jids = [model.joint(n).id for n in controller.joint_names]
    data.qpos[model.jnt_qposadr[jids]] = controller.default_pose
    mujoco.mj_forward(model, data)

    def tick(cmd):
        controller.set_twist(*cmd)
        controller.policy_step()
        for _ in range(duck_model.DECIMATION):
            controller.actuator_step()
            mujoco.mj_step(model, data)

    for _ in range(50):  # 1 s standing
        tick((0.0, 0.0, 0.0))
    for _ in range(50):  # 1 s to get into the gait
        tick(command)
    trunk = data.body('agent')
    # Integrate per tick: a single start/end yaw difference wraps once the duck turns > pi.
    forward_dist, yaw_total, fallen = 0.0, 0.0, False
    pos, yaw = trunk.xpos[:2].copy(), np.arctan2(trunk.xmat[3], trunk.xmat[0])
    for _ in range(int(seconds * 50)):
        tick(command)
        new_pos, new_yaw = trunk.xpos[:2].copy(), np.arctan2(trunk.xmat[3], trunk.xmat[0])
        forward_dist += float(np.dot(new_pos - pos, [np.cos(yaw), np.sin(yaw)]))
        yaw_total += float(np.angle(np.exp(1j * (new_yaw - yaw))))
        pos, yaw = new_pos, new_yaw
        fallen |= trunk.xmat[8] < 0.5
    return forward_dist / seconds, yaw_total / seconds, fallen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=4.0)
    args = parser.parse_args()
    controller = WalkingController(paths.policy_path())
    print(f'{"command (vx, vy, wz)":>24} | {"measured vx m/s":>15} | {"measured wz rad/s":>17} | fell')
    for cmd in [(0.1, 0, 0), (0.2, 0, 0), (0.25, 0, 0), (0.3, 0, 0), (0.35, 0, 0), (0.4, 0, 0), (0, 0, 0.5), (0, 0, 1.0)]:
        vx, wz, fell = run(controller, cmd, args.seconds)
        print(f'{cmd!s:>24} | {vx:15.3f} | {wz:17.3f} | {fell}')


if __name__ == '__main__':
    main()
