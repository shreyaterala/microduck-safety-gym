"""Download the Microduck model and walking policy ahead of time.

Clones pollen-robotics/microduck_rl at the pinned commit into third_party/microduck_rl and
downloads velstand.onnx from pollen-robotics/microduck-policies into assets/policies. Both also
happen automatically on first use; run this to do it up front (e.g. before going offline).

    python scripts/fetch_assets.py
"""

from __future__ import annotations

from microduck_safety_gym import model, paths


def main() -> None:
    print('robot MJCF :', paths.robot_xml())
    print('policy     :', paths.policy_path())
    print('agent XML  :', model.build_agent_xml())


if __name__ == '__main__':
    main()
