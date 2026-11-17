"""Make OmniSafe accept the Microduck envs.

OmniSafe's ``SafetyGymnasiumEnv`` (the CMDP wrapper behind its Safety Gymnasium benchmarks)
only accepts env ids from a hard-coded list. This subclass is that same class with the
Microduck ids, so PPO, PPO-Lag, CPO, etc. train on the duck through exactly the code path they
use for Point and Car. Import it before calling ``omnisafe.Agent``:

    import microduck_safety_gym.omnisafe_env  # noqa: F401
"""

from __future__ import annotations

from typing import ClassVar

import microduck_safety_gym  # imports safety_gymnasium with the 3.12 fix, registers envs
from omnisafe.envs.core import env_register
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv


@env_register
class MicroduckSafetyGymnasiumEnv(SafetyGymnasiumEnv):
    """OmniSafe CMDP for ``SafetyMicroduck*-v0`` (vector observations)."""

    _support_envs: ClassVar[list[str]] = [
        env_id for env_id in microduck_safety_gym.ENV_IDS if 'Vision' not in env_id
    ]
