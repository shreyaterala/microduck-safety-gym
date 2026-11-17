"""Make safety-gymnasium 1.0.0 importable on Python >= 3.11.

safety-gymnasium declares dataclass fields such as ``color: np.array = COLOR['hazard']``.
Python 3.11 started rejecting unhashable dataclass defaults, so ``import safety_gymnasium``
raises ``ValueError: mutable default <class 'numpy.ndarray'>``. Upstream ``main`` still has
the bug.

While safety_gymnasium is being imported (and only then), this module wraps
``dataclasses._get_field`` so that a NumPy-array default becomes
``field(default_factory=<copy of that array>)``. That is the same fix people apply by hand,
it gives each instance its own array, and nothing outside that import is affected.

``microduck_safety_gym`` calls :func:`import_safety_gymnasium` before anything else, so
import this package before ``safety_gymnasium``.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np


def _patched(orig):
    def _get_field(cls, a_name, a_type, *args, **kwargs):
        default = cls.__dict__.get(a_name, dataclasses.MISSING)
        if isinstance(default, np.ndarray):
            setattr(cls, a_name, dataclasses.field(default_factory=default.copy))
        return orig(cls, a_name, a_type, *args, **kwargs)

    return _get_field


def _patch_gymnasium_markers() -> None:
    """Fix rendering under MuJoCo 3 in gymnasium 0.28.1 (pinned by safety-gymnasium).

    ``BaseRender._add_marker_to_scene`` sets ``MjvGeom.texid`` and other fields that MuJoCo 3
    removed, so every ``render()`` raises AttributeError. This replacement initializes the geom
    with ``mjv_initGeom`` (what newer gymnasium releases do) and then applies the marker fields.
    """
    import mujoco  # noqa: PLC0415
    from gymnasium.envs.mujoco import mujoco_rendering  # noqa: PLC0415

    def _add_marker_to_scene(self, marker: dict) -> None:
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f'Ran out of geoms. maxgeom: {self.scn.maxgeom}')
        geom = self.scn.geoms[self.scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.ones(3) * 0.1,
            np.zeros(3),
            np.eye(3).flatten(),
            np.ones(4, dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco.mjtGeom)):
                setattr(geom, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(geom, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == 'label', 'Only label is a string in mjvGeom.'
                geom.label = value
            elif hasattr(geom, key):
                raise ValueError(f'mjvGeom has attr {key} but type {type(value)} is invalid')
            else:
                raise ValueError(f"mjvGeom doesn't have field {key}")
        self.scn.ngeom += 1

    mujoco_rendering.BaseRender._add_marker_to_scene = _add_marker_to_scene  # noqa: SLF001


def import_safety_gymnasium():
    """Import and return ``safety_gymnasium``, applying the compatibility fixes if needed."""
    if 'safety_gymnasium' in sys.modules:
        return sys.modules['safety_gymnasium']
    orig = dataclasses._get_field  # noqa: SLF001 - private, but stable across 3.11-3.12
    dataclasses._get_field = _patched(orig)  # noqa: SLF001
    try:
        import safety_gymnasium  # noqa: PLC0415
    finally:
        dataclasses._get_field = orig  # noqa: SLF001
    _patch_gymnasium_markers()
    return safety_gymnasium
