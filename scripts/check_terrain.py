"""How far Pollen's pretrained walking policy (velstand) goes off flat ground, with no retraining.

Each scenario runs the policy in a plain MuJoCo scene (BAM actuators, 50 Hz, as in
check_gait.py) for a few trials and reports speed, sideways drift, falls, and whether the duck
got back up after a fall (zero command for 6 s afterwards):

- friction: every contact's sliding friction set to mu (default 1.0)
- bumps:    random 3-6 cm square blocks of a given height covering ~30% of the floor
- slope:    the floor tilted by an angle; walk straight uphill and downhill
- ledge:    start on a platform 15 cm from its edge and walk off a drop of a given height
- step:     walk into a step up of a given height

    python scripts/check_terrain.py            # all scenarios
    python scripts/check_terrain.py --only friction ledge
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np

from microduck_safety_gym import model as duck_model
from microduck_safety_gym import paths
from microduck_safety_gym.controller import WalkingController

FORWARD = (0.4, 0.0, 0.0)  # full forward command (achieves ~0.1-0.13 m/s on flat ground)
TURN = (0.0, 0.0, 1.0)


@dataclass
class Result:
    speed: float  # m/s along the initial heading
    drift: float  # m, max sideways distance from the initial line
    yaw_rate: float  # rad/s
    fell: bool
    recovered: bool | None  # None when it did not fall


def make_scene(scenario: str, value: float, rng: np.random.Generator):
    spec = mujoco.MjSpec.from_file(str(duck_model.build_agent_xml()))
    world = spec.worldbody
    start_z = 0.0
    if scenario == 'slope':
        a = np.deg2rad(value)  # rotate about y: +x goes uphill for value > 0
        quat = [np.cos(-a / 2), 0, np.sin(-a / 2), 0]
        world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], quat=quat)
    else:
        world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    if scenario == 'bumps':
        h = value / 100
        for i in range(250):
            x, y = rng.uniform(-0.3, 2.0), rng.uniform(-0.8, 0.8)
            if abs(x) < 0.12 and abs(y) < 0.12:
                continue  # keep the spawn spot clear
            s = rng.uniform(0.015, 0.03)
            world.add_geom(name=f'bump{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, h / 2],
                           pos=[x, y, h / 2])
    if scenario == 'ledge':
        h = value / 100  # platform from x=-1 to x=+0.15, top at height h
        world.add_geom(name='platform', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.575, 1.0, h / 2],
                       pos=[-0.425, 0, h / 2])
        start_z = h
    if scenario == 'step':
        h = value / 100  # step up starting 15 cm ahead
        world.add_geom(name='step', type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.0, 1.0, h / 2],
                       pos=[1.15, 0, h / 2])
    model = spec.compile()
    if scenario == 'friction':
        model.geom_friction[:, 0] = value  # MuJoCo uses the max of the pair, so set every geom
    return model, mujoco.MjData(model), start_z


def run(controller, scenario, value, command, seconds, seed) -> Result:
    rng = np.random.default_rng(seed)
    model, data, start_z = make_scene(scenario, value, rng)
    # Bind first: BAM's controller calls mj_setConst, which leaves data.qpos at qpos0.
    controller.bind(model, data)
    free = model.joint('trunk_base_freejoint').qposadr[0]
    data.qpos[free + 2] += start_z
    yaw = rng.uniform(-0.1, 0.1)  # small heading jitter between trials
    data.qpos[free + 3:free + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    jids = [model.joint(n).id for n in controller.joint_names]
    data.qpos[model.jnt_qposadr[jids]] = controller.default_pose
    mujoco.mj_forward(model, data)
    trunk = data.body('agent')

    def tick(cmd):
        controller.set_twist(*cmd)
        controller.policy_step()
        for _ in range(duck_model.DECIMATION):
            controller.actuator_step()
            mujoco.mj_step(model, data)

    def down():
        return trunk.xmat[8] < 0.5

    for _ in range(50):  # 1 s standing
        tick((0.0, 0.0, 0.0))
    p0 = trunk.xpos[:2].copy()
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-heading[1], heading[0]])
    drift, fell = 0.0, down()
    yaw_prev, yaw_total, ticks = np.arctan2(trunk.xmat[3], trunk.xmat[0]), 0.0, 0
    for _ in range(int(seconds * 50)):
        tick(command)
        ticks += 1
        drift = max(drift, abs(float(np.dot(trunk.xpos[:2] - p0, normal))))
        y = np.arctan2(trunk.xmat[3], trunk.xmat[0])
        yaw_total += float(np.angle(np.exp(1j * (y - yaw_prev))))
        yaw_prev = y
        if down():
            fell = True
            break
    speed = float(np.dot(trunk.xpos[:2] - p0, heading)) / seconds
    recovered = None
    if fell:
        for _ in range(300):  # 6 s at zero command: does velstand get back up?
            tick((0.0, 0.0, 0.0))
        recovered = bool(trunk.xmat[8] > 0.8)
    return Result(speed, drift, yaw_total / (ticks * 0.02), fell, recovered)


def report(name, results):
    falls = sum(r.fell for r in results)
    rec = [r.recovered for r in results if r.fell]
    rec_s = f'{sum(rec)}/{len(rec)} got up' if rec else ''
    speeds = [r.speed for r in results if not r.fell]
    sp = f'{np.mean(speeds):6.3f}' if speeds else '   -  '
    if 'turn' in name:
        wz = [r.yaw_rate for r in results if not r.fell]
        sp += f' m/s  turn {np.mean(wz):5.2f} rad/s' if wz else ' m/s  turn   -  rad/s'
        sp = sp.replace(' m/s  turn', '  turn', 0)
    print(f'  {name:28s} speed {sp} m/s  drift {np.mean([r.drift for r in results]) * 100:5.1f} cm  '
          f'falls {falls}/{len(results)}  {rec_s}', flush=True)


SCENARIOS = {
    'friction': [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],  # mu
    'bumps': [0.5, 1.0, 2.0, 3.0],  # cm
    'slope': [3.0, 6.0, 10.0, 15.0],  # degrees
    'ledge': [1.0, 2.0, 4.0, 6.0, 10.0],  # cm drop
    'step': [0.5, 1.0, 2.0, 3.0],  # cm rise
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', choices=list(SCENARIOS))
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    controller = WalkingController(paths.policy_path())
    seeds = range(args.trials)
    for scenario in args.only or SCENARIOS:
        print(scenario, flush=True)
        for v in SCENARIOS[scenario]:
            if scenario == 'friction':
                report(f'mu={v} forward', [run(controller, scenario, v, FORWARD, 6, s) for s in seeds])
                report(f'mu={v} turn in place', [run(controller, scenario, v, TURN, 4, s) for s in seeds])
            elif scenario == 'slope':
                report(f'{v:g} deg uphill', [run(controller, scenario, v, FORWARD, 6, s) for s in seeds])
                report(f'{v:g} deg downhill', [run(controller, scenario, -v, FORWARD, 6, s) for s in seeds])
            else:
                secs = 8 if scenario == 'bumps' else 6
                report(f'{v:g} cm', [run(controller, scenario, v, FORWARD, secs, s) for s in seeds])


if __name__ == '__main__':
    main()
