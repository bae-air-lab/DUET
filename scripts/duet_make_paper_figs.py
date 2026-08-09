"""Regenerate the two data figures at single-column print size.

Both are drawn at their final width on the page (6.5 in) with a 10 pt base
font, so the type in the figure matches the type in the document. Drawing
them larger and letting Word scale them down is what made the old versions
unreadable.

Usage:
  python scripts/duet_make_paper_figs.py
"""

from __future__ import annotations

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402

REPO = "/home/rbbist.lab/unitree_rl_mjlab"
FIGS = os.path.join(REPO, "medias/paper_figs")
WIDTH_IN = 6.5  # the text width of the single-column layout

plt.rcParams.update({
  "font.family": "DejaVu Sans",
  "font.size": 10,
  "axes.titlesize": 10,
  "axes.labelsize": 10,
  "xtick.labelsize": 10,
  "ytick.labelsize": 10,
  "legend.fontsize": 10,
  "axes.linewidth": 0.8,
  "figure.dpi": 300,
})

# Soft highlighter tones rather than saturated primaries: at this panel size
# the fully saturated set read as harsh.
REF = "Abl-Reference"
REF_COLOUR = "#7FB3D5"
VARIANTS = [("Abl-NoSymmetry", "no symmetry", "#F1948A"),
            ("Abl-NoHeightCmd", "no height command", "#A9DFA0"),
            ("Abl-NoArmCurriculum", "no arm curriculum", "#C9AEDD")]
METRICS = [("lin_vel_err", "Lin. vel. error (m/s) ↓", False),
           ("ang_vel_err", "Ang. vel. error (rad/s) ↓", False),
           ("height_err", "Height error (m) ↓", False),
           ("symmetry_loss", "Symmetry loss ↓", True),
           ("living_time", "Living time (s) ↑", False)]


def ablation_curves():
  rows = [json.loads(l) for l in
          open(os.path.join(REPO, "logs/duet_runs/ablation_curves.jsonl"))]
  by = collections.defaultdict(dict)
  for r in rows:
    by[r["variant"]][r["iteration"]] = r

  fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 4.3))
  axes = axes.ravel()
  for ax, (key, title, logy) in zip(axes, METRICS):
    its = sorted(by[REF])
    # Lighter colours need a little more weight to stay visible on white.
    ax.plot(its, [by[REF][i][key] for i in its], "o-", color=REF_COLOUR,
            lw=2.4, ms=5.0, label="full method", zorder=5)
    for name, label, colour in VARIANTS:
      if name not in by:
        continue
      v = sorted(by[name])
      ax.plot(v, [by[name][i][key] for i in v], "s--", color=colour,
              lw=2.2, ms=4.5, label=label)
    if logy:
      ax.set_yscale("log")
    ax.set_title(title, pad=4)
    ax.set_xlabel("iteration", labelpad=1)
    ax.set_xticks([1000, 3000, 5000])
    ax.tick_params(length=2.5, pad=1.5)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
      ax.spines[sp].set_visible(False)

  # Sixth cell carries the legend, so no panel has to give up space for it.
  legend = axes[-1]
  legend.axis("off")
  handles, labels = axes[0].get_legend_handles_labels()
  legend.legend(handles, labels, loc="center", frameon=False,
                handlelength=2.2, labelspacing=0.9)

  fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.2)
  out = os.path.join(FIGS, "fig_ablation_curves.png")
  fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
  plt.close(fig)
  print("wrote", out, plt.imread(out).shape)


def _soft(name, deep):
  """A near-white to light-tint ramp. Keeps the heat maps low-contrast."""
  return matplotlib.colors.LinearSegmentedColormap.from_list(
    name, ["#fdfdfd", deep[0], deep[1]])


PANELS = [("falls_per_env_min", "Falls per env-minute", "{:.2f}",
           _soft("soft_green", ("#eef8f1", "#b9e2c9"))),
          ("living_s", "Living time (s)", "{:.1f}",
           _soft("soft_violet", ("#f5f1fa", "#d2c5ea"))),
          ("vxy_err", "Linear velocity error (m/s)", "{:.2f}",
           _soft("soft_blue", ("#eef4fc", "#bcd8f3")))]

# The colour ramps are light throughout, so the annotations are dark rather
# than the white used when these panels were drawn on viridis.
CELL_TEXT = "#243044"


def envelope():
  rows = [json.loads(l) for l in
          open(os.path.join(REPO, "logs/duet_runs/robustness_envelope.jsonl"))]
  pays = sorted({r["payload"] for r in rows})
  pushes = sorted({r["push"] for r in rows})
  grid = {(r["payload"], r["push"]): r for r in rows}

  # Stacked rather than side by side: at 6.5 in a row of three heat maps
  # leaves each cell too narrow to hold a 10 pt number.
  fig, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, 6.4))
  for ax, (key, title, fmt, cmap) in zip(axes, PANELS):
    m = np.array([[grid[(p, k)][key] if (p, k) in grid else np.nan
                   for k in pushes] for p in pays], dtype=float)
    im = ax.imshow(m, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pushes)), [f"{p:g}" for p in pushes])
    ax.set_yticks(range(len(pays)), [f"{p:g}" for p in pays])
    ax.set_xlabel("push (m/s)", labelpad=2)
    ax.set_ylabel("payload (kg/hand)", labelpad=2)
    ax.set_title(title, pad=5)
    ax.tick_params(length=0, pad=2)
    for i in range(len(pays)):
      for j in range(len(pushes)):
        if np.isnan(m[i, j]):
          continue
        ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                color=CELL_TEXT, fontsize=10)
    # Dashed line marking the payload ceiling seen in training.
    ceiling = 1.75
    if ceiling in pays:
      ax.axhline(pays.index(ceiling) + 0.5, color="#5a6472", ls="--", lw=1.4)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.030)
    cb.ax.tick_params(labelsize=10, length=2, pad=2)

  fig.tight_layout(pad=0.4, h_pad=1.4)
  out = os.path.join(FIGS, "fig_envelope.png")
  fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
  plt.close(fig)
  print("wrote", out, plt.imread(out).shape)


if __name__ == "__main__":
  ablation_curves()
  envelope()
