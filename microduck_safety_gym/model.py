"""Turn Pollen's Microduck MJCF into a Safety Gymnasium agent XML.

Safety Gymnasium builds every world from the agent's XML (``World.build``): it expects one
top-level body named ``agent`` carrying a site named ``agent`` with the standard sensors
(accelerometer, velocimeter, gyro, magnetometer), adds the floor, goal, hazards and vases
around it, and overwrites ``compiler/meshdir``. This module generates that XML from
``robot_allcollisions.xml``:

- trunk body ``trunk_base`` -> ``agent``; adds the ``agent`` site, the four sensors (same as
  Point/Car) and a forward-facing ``vision`` camera for the Vision task variants;
- mesh paths made absolute, so Safety Gymnasium's meshdir override does not break them;
- geoms named (``duck/<i>``), so contact costs can attribute contacts to the robot;
- position actuators converted to BAM torque motors exactly as Pollen's CPU rehearsal
  (``scripts/infer_policy.py``) and training (``bam.mjlab.BamActuator``) do;
- physics timestep 5 ms (the policy runs every 4 steps = 50 Hz, as in training).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from microduck_safety_gym import paths

PHYSICS_DT = 0.005
DECIMATION = 4  # physics steps per policy tick -> 50 Hz control
SPAWN_HEIGHT = 0.12  # trunk height of the STAND keyframe

# BAM M6 model of the Dynamixel XL330, with the settings Pollen trains and rehearses with
# (microduck_rl/scripts/infer_policy.py defaults).
BAM_MOTOR, BAM_MODEL = 'xl330', 'm6'
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

# Sensors Safety Gymnasium agents observe by default (BaseAgent.SensorConf.sensors).
AGENT_SENSORS = ('accelerometer', 'velocimeter', 'gyro', 'magnetometer')
# Sensors the walking policy reads; everything else in the source MJCF is dropped.
KEEP_SENSORS = ('imu_ang_vel', 'imu_accel')

_GENERATOR_VERSION = '1'


def load_bam_model():
    """The BAM XL330 model with Pollen's firmware gain and nominal battery voltage."""
    from bam.model import load_model  # noqa: PLC0415

    model = load_model(motor_name=BAM_MOTOR, model=BAM_MODEL)
    model.actuator.kp = BAM_KP_FW
    model.actuator.vin = BAM_VIN
    model.actuator.max_current = None  # training runs without the firmware current limiter
    return model


def _build_spec(source: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(source))
    spec.option.timestep = PHYSICS_DT

    meshdir = (source.parent / spec.meshdir).resolve()
    for mesh in spec.meshes:
        mesh.file = (meshdir / mesh.file).as_posix()
    spec.meshdir = ''

    trunk = spec.body('trunk_base')
    trunk.name = 'agent'
    trunk.pos = [0.0, 0.0, SPAWN_HEIGHT]
    trunk.add_site(name='agent', pos=[0, 0, 0], size=[0.02, 0, 0], rgba=[1, 0, 0, 0.1], group=3)
    trunk.add_camera(name='vision', pos=[0.06, 0, 0.06], xyaxes=[0, -1, 0, 0.4, 0, 1], fovy=90)
    for exclude in spec.excludes:
        if exclude.bodyname1 == 'trunk_base':
            exclude.bodyname1 = 'agent'
        if exclude.bodyname2 == 'trunk_base':
            exclude.bodyname2 = 'agent'

    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f'duck/{i}'

    for sensor in list(spec.sensors):
        if sensor.name not in KEEP_SENSORS:
            spec.delete(sensor)
    for name, kind in zip(
        AGENT_SENSORS,
        (
            mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            mujoco.mjtSensor.mjSENS_VELOCIMETER,
            mujoco.mjtSensor.mjSENS_GYRO,
            mujoco.mjtSensor.mjSENS_MAGNETOMETER,
        ),
    ):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='agent')

    # Position actuators -> torque motors driven by BAM (mirrors infer_policy.load_mujoco_with_bam).
    bam = load_bam_model()
    force_limit = bam.actuator.vin * bam.kt.value / bam.R.value
    joints = {j.name: j for j in spec.joints}
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = [-force_limit, force_limit]
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        joint = joints[act.target]
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = 0.0
        joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
        joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
    return spec


def build_agent_xml() -> Path:
    """Write (or reuse) the generated agent XML and return its absolute path."""
    source = paths.robot_xml()
    key = hashlib.sha1(  # noqa: S324 - cache key, not security
        source.read_bytes() + _GENERATOR_VERSION.encode() + str(source.resolve()).encode(),
    ).hexdigest()[:12]
    out = paths.CACHE_DIR / f'microduck_agent_{key}.xml'
    if not out.is_file():
        spec = _build_spec(source)
        spec.compile()  # fail here, not deep inside Safety Gymnasium
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(spec.to_xml(), encoding='utf-8')
    return out
