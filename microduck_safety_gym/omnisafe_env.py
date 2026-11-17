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


def load_evaluator(run_dir, model_name: str, env_id: str | None = None, **render_kwargs):
    """``omnisafe.Evaluator`` for a saved checkpoint, optionally on a different env.

    OmniSafe always rebuilds the env a run was trained on (from its ``config.json``). Passing
    ``env_id`` deploys the same policy, with the same observation normalizer, on another env
    instead, e.g. a policy trained on ``SafetyMicroduckGoal1-v0`` evaluated on
    ``SafetyMicroduckGoal1MismatchSevere-v0`` (train nominal, deploy mismatched). The two envs
    must share observation and action spaces, which every Microduck Goal variant does.
    """
    import omnisafe  # local: this module must be importable before omnisafe

    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    if env_id is not None:
        load_cfgs = evaluator._Evaluator__load_cfgs  # noqa: SLF001  (no public override)

        def load_cfgs_with_env(save_dir: str) -> None:
            load_cfgs(save_dir)
            evaluator._cfgs['env_id'] = env_id  # noqa: SLF001
            evaluator._dict_cfgs['env_id'] = env_id  # noqa: SLF001

        evaluator._Evaluator__load_cfgs = load_cfgs_with_env  # noqa: SLF001
    evaluator.load_saved(save_dir=str(run_dir), model_name=model_name, **render_kwargs)
    return evaluator
