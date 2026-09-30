"""Terrain sub-generators shared by the task packages.

mjlab's ``HfRandomUniformTerrainCfg`` ignores the difficulty it is handed
(``del difficulty`` in ``mjlab/terrains/heightfield_terrains.py``), so a
terrain curriculum over it would be equally rough on every row. The subclass
here scales the noise with difficulty instead. It lives here, not in mjlab,
because mjlab is a pinned dependency.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import HfRandomUniformTerrainCfg
from mjlab.terrains.terrain_generator import TerrainOutput

__all__ = ["HfScaledRandomUniformTerrainCfg"]


@dataclass(kw_only=True)
class HfScaledRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
  """``HfRandomUniformTerrainCfg`` whose noise grows with difficulty.

  At difficulty ``t`` the upper noise bound is
  ``noise_range[0] + t * (noise_range[1] - noise_range[0])``; with
  ``noise_range[0] = 0`` the surface is exactly flat at ``t = 0``.
  """

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    lo, hi = self.noise_range
    upper = lo + difficulty * (hi - lo)
    # The parent converts bounds to heightfield units with int(), which
    # truncates float quotients such as 0.015 / 0.005 = 2.9999999999999996 to
    # 2; the epsilon keeps a bound that is a whole number of units exact.
    scaled = dataclasses.replace(self, noise_range=(lo, upper + 1e-9))
    return HfRandomUniformTerrainCfg.function(scaled, difficulty, spec, rng)
