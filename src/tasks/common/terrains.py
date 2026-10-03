"""Sub-terrains not provided by mjlab.

``SoftFlatTerrainCfg`` is a flat tile whose CONTACT is soft, standing in for
deformable ground such as poultry litter: the feet sink in, the ground pushes
back sluggishly, and it is slippery. The geometry is mjlab's flat box; only
the contact parameters differ.

Why ``priority``: in MuJoCo, when two geoms touch and their ``priority``
differs, the higher-priority geom's condim, friction, solref and solimp are
used for that contact and the other geom's are ignored. The G1 foot geoms are
priority 1 (``g1_23dof_constants.FULL_COLLISION``), which is what makes the
per-env foot friction randomisation apply on every tile. A soft tile at the
default priority 0 would therefore lose to the feet and behave like ordinary
hard ground; at priority 2 it wins on this tile only. The side effect is that
the tile's own friction replaces the foot friction here, so it is sampled per
tile from ``friction_range``.

Softness is scaled by the curriculum difficulty (the terrain row): at
difficulty 0 solref/solimp equal MuJoCo's defaults, i.e. exactly today's hard
ground, and they move linearly to the soft end at difficulty 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane

__all__ = ["SoftFlatTerrainCfg"]


def _lerp(lo_hi: tuple[float, float], t: float) -> float:
  return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * t


@dataclass(kw_only=True)
class SoftFlatTerrainCfg(SubTerrainCfg):
  """Flat tile with soft, difficulty-scaled contact. Each range is (hard, soft)."""

  # solref = (timeconst, dampratio). Larger timeconst = softer spring; a
  # dampratio above 1 is overdamped, so the ground gives without springing
  # back. timeconst must stay >= 2x the 0.005 s physics timestep.
  timeconst_range: tuple[float, float] = (0.02, 0.05)
  dampratio_range: tuple[float, float] = (1.0, 2.0)
  # solimp = (dmin, dmax, width, 0.5, 2). Impedance rises from dmin at the
  # surface to dmax at ``width`` of penetration: soft on first touch, firmer as
  # the foot sinks, like litter compacting. Lower impedance = deeper sinkage.
  dmin_range: tuple[float, float] = (0.9, 0.6)
  dmax_range: tuple[float, float] = (0.95, 0.9)
  width_range: tuple[float, float] = (0.001, 0.02)
  # Sliding friction, drawn uniformly per tile (not scaled by difficulty).
  friction_range: tuple[float, float] = (0.3, 1.0)
  # Must exceed the foot geoms' priority (1), see the module docstring.
  priority: int = 2

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    t = float(np.clip(difficulty, 0.0, 1.0))
    body = spec.body("terrain")
    origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
    box = make_plane(body, self.size, 0.0, center_zero=False)[0]
    box.priority = self.priority
    box.solref = (_lerp(self.timeconst_range, t), _lerp(self.dampratio_range, t))
    box.solimp = (
      _lerp(self.dmin_range, t),
      _lerp(self.dmax_range, t),
      _lerp(self.width_range, t),
      0.5,
      2.0,
    )
    box.friction = (float(rng.uniform(*self.friction_range)), 0.005, 0.0001)
    # Litter brown, darker on softer tiles, so they read in the viewer.
    shade = 0.75 - 0.35 * t
    color = (shade, 0.8 * shade, 0.55 * shade, 1.0)
    return TerrainOutput(
      origin=np.array(origin), geometries=[TerrainGeometry(geom=box, color=color)]
    )
