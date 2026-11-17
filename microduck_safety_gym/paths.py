"""Where the Microduck model and walking policy live, and how to fetch them.

Neither is vendored in this repo:

- The robot MJCF comes from ``pollen-robotics/microduck_rl`` (MJCF: Apache-2.0; meshes:
  CC BY-NC-SA 4.0, see that repo's LICENSE-MODELS.md). It is cloned into
  ``third_party/microduck_rl`` at a pinned commit.
- The walking policy is ``velstand.onnx`` from the public Hugging Face repo
  ``pollen-robotics/microduck-policies``, downloaded into ``assets/policies``.

Both are fetched on first use, or ahead of time with ``python scripts/fetch_assets.py``.
Set ``MICRODUCK_RL_DIR`` / ``MICRODUCK_POLICY_DIR`` to use copies stored elsewhere.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / '.cache'

MICRODUCK_RL_URL = 'https://github.com/pollen-robotics/microduck_rl.git'
# microduck_rl commit the model was taken from (2026-10-27).
MICRODUCK_RL_COMMIT = 'bae39f5900b318f7e87fc485c30a31213cfa14eb'

POLICY_REPO = 'pollen-robotics/microduck-policies'
# The shipped default walk: tracks (vx, vy, yaw rate), stands still at zero command and gets back
# up after a fall. Trained on Mjlab-VelStand-Rough-Backlash-MicroDuck with the all-collisions model.
DEFAULT_POLICY = 'velstand.onnx'

_ROBOT_XML_REL = Path('src/mjlab_microduck/robot/microduck/robot_allcollisions.xml')


def microduck_rl_dir(fetch: bool = True) -> Path:
    """Root of the microduck_rl checkout, cloning it at the pinned commit if missing."""
    root = Path(os.environ.get('MICRODUCK_RL_DIR', REPO_ROOT / 'third_party' / 'microduck_rl'))
    if not (root / _ROBOT_XML_REL).is_file():
        if not fetch:
            raise FileNotFoundError(f'microduck_rl not found at {root}; run scripts/fetch_assets.py')
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', MICRODUCK_RL_URL, str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', '--quiet', MICRODUCK_RL_COMMIT], check=True)
    return root


def robot_xml(fetch: bool = True) -> Path:
    """The all-collisions Microduck MJCF (the model velstand was trained on)."""
    return microduck_rl_dir(fetch) / _ROBOT_XML_REL


def policy_path(name: str = DEFAULT_POLICY, fetch: bool = True) -> Path:
    """Local path of a policy from pollen-robotics/microduck-policies, downloading it if missing."""
    folder = Path(os.environ.get('MICRODUCK_POLICY_DIR', REPO_ROOT / 'assets' / 'policies'))
    path = folder / name
    if not path.is_file():
        if not fetch:
            raise FileNotFoundError(f'{path} missing; run scripts/fetch_assets.py')
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        folder.mkdir(parents=True, exist_ok=True)
        hf_hub_download(POLICY_REPO, name, local_dir=folder)
    return path
