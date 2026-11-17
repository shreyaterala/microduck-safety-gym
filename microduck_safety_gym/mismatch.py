"""Mismatch knobs: ways the duck's motion departs from the commands the RL agent sends.

Each knob widens the gap between the commanded (vx, yaw rate) and what the robot does, on top
of the walking policy's own dead zone and lag. They exist to test safety methods whose
guarantees assume an accurate motion model (e.g. a CBF filter on a unicycle model).

- **Command delay:** the agent's command reaches the walking policy ``command_delay_steps``
  env steps (20 ms each) late.
- **Battery sag:** the servo model's supply voltage (nominal 7.4 V). Lower voltage means less
  torque; BAM's load-dependent voltage drop stays on, with the same 1.4 V headroom.
- **Shoves:** a horizontal push on the trunk, ``shove_force`` newtons for ``shove_duration``
  seconds, in a random direction, every ``shove_interval`` seconds (drawn uniformly).
- **Floor friction:** the sliding friction between the duck and the floor (default 1.0). The
  feet slip, so turns and straight lines come out short or crooked.

Levels are registered as env ids, e.g. ``SafetyMicroduckGoal1MismatchSevere-v0``.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Mismatch:
    command_delay_steps: int = 0  # env steps of 20 ms
    battery_vin: float = 7.4  # V
    shove_force: float = 0.0  # N (the duck weighs ~0.8 kg)
    shove_duration: float = 0.1  # s
    shove_interval: tuple[float, float] = (2.0, 4.0)  # s, uniform
    floor_friction: float | None = None  # sliding mu; None = unchanged (1.0)
    # If set, each episode draws its supply voltage uniformly from these values instead of
    # using battery_vin (the 50/50 nominal/reduced-strength training mix of Q3).
    battery_vin_choices: tuple[float, ...] | None = None

    @property
    def shoves(self) -> bool:
        return self.shove_force > 0.0


# Shove forces were calibrated by measurement: the walking policy absorbs pushes up to about
# 2.5 N x 0.1 s almost completely, while at 3 N with 100 ms delay and 6.0 V it falls in about half
# of 20 s episodes and at 4 N in nearly all. Severe is set just below the falling regime
# (~9 cm of standing drift per 20 s, ~1 fall in 10 episodes), so it perturbs the motion without
# mostly ending episodes early.
# Friction likewise: on its own the duck walks down to mu = 0.2 (turning in place drops to
# ~0.2 rad/s), but combined with the severe delay, sag and shoves it fell in 3/6 episodes at
# mu = 0.25 and 6/6 at mu = 0.2, versus 0/6 at mu = 0.3 (turn rate 0.76 rad/s for a 1.0 command).
LEVELS: dict[str, Mismatch] = {
    'none': Mismatch(),
    'moderate': Mismatch(
        command_delay_steps=2, battery_vin=6.6, shove_force=2.0, floor_friction=0.4
    ),  # 40 ms
    'severe': Mismatch(
        command_delay_steps=5, battery_vin=6.0, shove_force=2.5, floor_friction=0.3
    ),  # 100 ms
    # Single-knob ablation: each knob alone at its severe value, everything else nominal.
    'delay': Mismatch(command_delay_steps=5),
    'battery': Mismatch(battery_vin=6.0),  # "reduced motor strength" in Q3
    'shoves': Mismatch(shove_force=2.5),
    'friction': Mismatch(floor_friction=0.3),
    # Q3 training mix: half the episodes at 7.4 V (nominal), half at 6.0 V (reduced).
    'batterymix': Mismatch(battery_vin_choices=(7.4, 6.0)),
}
