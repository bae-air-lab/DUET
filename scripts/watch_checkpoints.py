#!/usr/bin/env python3
"""Watch a training run and report each new checkpoint with its key metrics.

Polls the newest run directory under the experiment folder for `model_*.pt`
files. When a new checkpoint at iteration N appears, it parses the training
stdout log for iteration N's rsl_rl summary block, appends a row of the main
metrics to a persistent markdown table (checkpoints_metrics.md in the run dir),
prints the full table, and exits (so the caller is re-invoked per checkpoint).

Usage: watch_checkpoints.py [TRAIN_LOG]   (default: logs/train_v2.log)
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path("logs/rsl_rl/LocoManip_G1_23dof")
TRAIN_LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/train_v2.log")
POLL_SECONDS = 30
TABLE_NAME = "checkpoints_metrics.md"
ITER_RE = re.compile(r"model_(\d+)\.pt$")

COLUMNS = ["#", "Iter", "Saved At", "MeanRew", "EpLen", "trk_height",
           "err_vxy", "err_vyaw", "fell_over"]
HEADER = "| " + " | ".join(COLUMNS) + " |\n" + "|" + "---|" * len(COLUMNS) + "\n"

# Log field label -> short key.
FIELDS = {
    "Mean reward": "MeanRew",
    "Mean episode length": "EpLen",
    "Episode_Reward/track_base_height": "trk_height",
    "Metrics/twist/error_vel_xy": "err_vxy",
    "Metrics/twist/error_vel_yaw": "err_vyaw",
    "Episode_Termination/fell_over": "fell_over",
}


def newest_run_dir():
    if not EXPERIMENT_DIR.exists():
        return None
    runs = [p for p in EXPERIMENT_DIR.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def checkpoints(run):
    ck = [p for p in run.glob("model_*.pt") if ITER_RE.search(p.name)]
    return sorted(ck, key=lambda p: int(ITER_RE.search(p.name).group(1)))


def parse_metrics_by_iter():
    """Return {iteration: {short_key: value}} parsed from the training stdout log."""
    if not TRAIN_LOG.exists():
        return {}
    blocks = {}
    cur = None
    for line in TRAIN_LOG.read_text(errors="ignore").splitlines():
        m = re.search(r"Learning iteration\s+(\d+)/", line)
        if m:
            cur = int(m.group(1))
            blocks.setdefault(cur, {})
            continue
        if cur is None:
            continue
        for label, key in FIELDS.items():
            mm = re.search(re.escape(label) + r":\s*(-?\d+\.?\d*)", line)
            if mm:
                blocks[cur][key] = float(mm.group(1))
    return blocks


def metrics_for(iteration, blocks):
    """Metrics for exactly `iteration`, else the nearest logged iteration <= it."""
    if iteration in blocks and blocks[iteration]:
        return blocks[iteration]
    below = [i for i in blocks if i <= iteration and blocks[i]]
    return blocks[max(below)] if below else {}


def fmt(v):
    return "-" if v is None else f"{v:.3f}".rstrip("0").rstrip(".")


def row(idx, ck, blocks):
    it = int(ITER_RE.search(ck.name).group(1))
    saved = datetime.fromtimestamp(ck.stat().st_mtime).strftime("%m-%d %H:%M:%S")
    m = metrics_for(it, blocks)
    cells = [str(idx), str(it), saved,
             fmt(m.get("MeanRew")), fmt(m.get("EpLen")), fmt(m.get("trk_height")),
             fmt(m.get("err_vxy")), fmt(m.get("err_vyaw")), fmt(m.get("fell_over"))]
    return "| " + " | ".join(cells) + " |\n"


def main():
    while newest_run_dir() is None:
        time.sleep(POLL_SECONDS)
    run = newest_run_dir()
    table = run / TABLE_NAME

    seen = set()
    if table.exists():
        for line in table.read_text().splitlines():
            m = re.match(r"\|\s*\d+\s*\|\s*(\d+)\s*\|", line)
            if m:
                seen.add(int(m.group(1)))
    else:
        table.write_text(HEADER)

    while True:
        run = newest_run_dir()
        new = [c for c in checkpoints(run)
               if int(ITER_RE.search(c.name).group(1)) not in seen]
        if new:
            blocks = parse_metrics_by_iter()
            existing = table.read_text() if table.exists() else HEADER
            # Count existing DATA rows (a row is "| <#> | <iter> | ...").
            n = sum(1 for l in existing.splitlines()
                    if re.match(r"\|\s*\d+\s*\|\s*\d+\s*\|", l))
            with table.open("a") as f:
                for c in new:
                    n += 1
                    f.write(row(n, c, blocks))
                    seen.add(int(ITER_RE.search(c.name).group(1)))
            print(f"Run: {run}")
            print(f"New checkpoint(s): {[c.name for c in new]}\n")
            print(table.read_text())
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
