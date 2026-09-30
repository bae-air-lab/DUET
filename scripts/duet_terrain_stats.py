"""Measure the litter terrain generator's surfaces (gate G6).

For every sub-terrain of ``litter_terrain_generator_cfg()`` at difficulty 0,
0.5 and 1, the surface is generated in isolation on one 8 x 8 m tile and
sampled by vertical ray casts on a ``--grid`` spacing. Reported per case:

  min / max / p2p   surface height extremes and peak-to-peak (m)
  max_slope         largest |grad z| from central differences over 2 x grid
                    (rise/run). For the box terrains (flat, clumps) the cell
                    edges are vertical faces, so this reads step / (2 x grid)
                    there; ``max_step`` is the meaningful number for them
  in_slope          max_slope inside the tile, excluding the flat border band
                    and one grid cell (mjlab's heightfield terrains do not taper
                    to their flat border, so the whole-tile maximum of an
                    undulating surface is the step at the border edge)
  max_step          largest height change between neighbouring samples (m)
  wavelength        2 x the mean spacing of mean-level crossings along rows and
                    columns, inside the tile border (m); the dominant horizontal
                    scale of an undulating surface

``--render`` additionally writes an offscreen MuJoCo render of the whole
generator grid (difficulty grows along x, one terrain type per column) and a
close-up of the hardest row.

Usage:
  PYTHONPATH=. python scripts/duet_terrain_stats.py
  PYTHONPATH=. python scripts/duet_terrain_stats.py --render logs/terrain_litter.png
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import mujoco
import numpy as np

from src.tasks.duet.config.g1_23dof.env_cfgs import litter_terrain_generator_cfg

DIFFICULTIES = (0.0, 0.5, 1.0)


def surface(sub_cfg, difficulty: float, size: tuple[float, float], grid: float,
            seed: int) -> np.ndarray:
  cfg = copy.deepcopy(sub_cfg)
  cfg.size = size
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg.function(difficulty, spec, np.random.default_rng(seed))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  xs = np.arange(grid / 2, size[0], grid)
  ys = np.arange(grid / 2, size[1], grid)
  z = np.empty((len(xs), len(ys)))
  top = 10.0
  down = np.array([0.0, 0.0, -1.0])
  geomid = np.zeros(1, dtype=np.int32)
  for i, x in enumerate(xs):
    for j, y in enumerate(ys):
      d = mujoco.mj_ray(model, data, np.array([x, y, top]), down, None, 1, -1, geomid)
      z[i, j] = top - d if d >= 0 else np.nan
  return z


def wavelength(z: np.ndarray, grid: float, border: float) -> float:
  b = int(np.ceil(border / grid)) + 1
  inner = z[b:-b, b:-b]
  if inner.size == 0 or np.ptp(inner) < 1e-6:
    return float("nan")
  c = inner - inner.mean()
  spacings = []
  for line in list(c) + list(c.T):
    s = np.signbit(line)
    idx = np.nonzero(s[1:] != s[:-1])[0]
    if len(idx) >= 2:
      spacings.extend(np.diff(idx) * grid)
  return 2.0 * float(np.mean(spacings)) if spacings else float("nan")


def render(path: str) -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  from mjlab.terrains.terrain_generator import TerrainGenerator

  cfg = litter_terrain_generator_cfg()
  cfg.seed = 0
  spec = mujoco.MjSpec()
  TerrainGenerator(cfg).compile(spec)
  spec.visual.global_.offwidth = 1920
  spec.visual.global_.offheight = 1080
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  renderer = mujoco.Renderer(model, height=1080, width=1920, max_geom=100_000)
  views = {
    "": dict(lookat=(0.0, 0.0, 0.0), distance=120.0, azimuth=180.0, elevation=-50.0),
    "_hardest_row": dict(lookat=(36.0, -20.0, 0.0), distance=18.0, azimuth=200.0,
                         elevation=-25.0),
  }
  root, ext = os.path.splitext(path)
  for suffix, v in views.items():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = v["lookat"]
    cam.distance, cam.azimuth, cam.elevation = v["distance"], v["azimuth"], v["elevation"]
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    out = f"{root}{suffix}{ext or '.png'}"
    import imageio.v3 as iio

    iio.imwrite(out, img)
    print(f"render -> {out}")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--grid", type=float, default=0.025)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--only", nargs="*", default=None, help="sub-terrain names")
  ap.add_argument("--render", default=None, help="write an offscreen render to this path")
  a = ap.parse_args()

  cfg = litter_terrain_generator_cfg()
  print(f"litter terrain: tile {cfg.size[0]} x {cfg.size[1]} m, {cfg.num_rows} rows x "
        f"{cfg.num_cols} cols, ray grid {a.grid} m, seed {a.seed}")
  print(f"{'sub-terrain':>12} {'diff':>5}{'min':>9} {'max':>8} {'p2p':>7} "
        f"{'max_slope':>9} {'in_slope':>8} {'max_step':>8} {'wavelength':>10}")
  for name, sub in cfg.sub_terrains.items():
    if a.only and name not in a.only:
      continue
    for d in DIFFICULTIES:
      mark = " "
      try:
        z = surface(sub, d, cfg.size, a.grid, a.seed)
      except ValueError:
        # mjlab's BoxRandomGrid (NaN colour ramp) and Perlin (zero-height
        # hfield) cannot build at EXACTLY 0; the generator draws row 0 from
        # U[0, 0.1), so evaluate just above it and mark the row with '*'.
        d, mark = 1e-4, "*"
        z = surface(sub, d, cfg.size, a.grid, a.seed)
      gx, gy = np.gradient(z, a.grid)
      g = np.hypot(gx, gy)
      slope = float(np.nanmax(g))
      b = int(np.ceil(getattr(sub, "border_width", 0.0) / a.grid)) + 2
      in_slope = float(np.nanmax(g[b:-b, b:-b]))
      step = float(max(np.nanmax(np.abs(np.diff(z, axis=0))),
                       np.nanmax(np.abs(np.diff(z, axis=1)))))
      wl = wavelength(z, a.grid, getattr(sub, "border_width", 0.0))
      print(f"{name:>12} {d:5.2f}{mark}{np.nanmin(z):8.4f} {np.nanmax(z):8.4f} "
            f"{np.nanmax(z) - np.nanmin(z):7.4f} {slope:9.3f} {in_slope:8.3f} {step:8.4f} "
            f"{wl:10.2f}")
  if a.render:
    render(a.render)
  return 0


if __name__ == "__main__":
  sys.exit(main())
