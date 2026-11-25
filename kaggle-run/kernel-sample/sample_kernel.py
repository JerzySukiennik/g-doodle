"""Draw with the trained G-Doodle and emit only SVGs.

CPU: a 16M model drawing a few hundred tokens does not need a GPU, and the
weekly GPU quota is spoken for. Rendering here rather than at home means 197 MB
of checkpoint stays on Kaggle and a few kilobytes of vector art come back.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-doodle.git"
BOOT = "/tmp/gd-sample"
OUT = "/kaggle/working"

CATEGORIES = ["cat", "dragon", "house", "guitar", "octopus",
              "piano", "helicopter", "bicycle", "face"]

subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)

hits = sorted(glob.glob("/kaggle/input/**/g-doodle.pt", recursive=True))
print("checkpointy:", hits, flush=True)
if not hits:
    raise SystemExit("brak g-doodle.pt w /kaggle/input")
ckpt = hits[0]

os.chdir(BOOT)
for cat in CATEGORIES:
    dest = f"{OUT}/{cat.replace(' ', '_')}.svg"
    r = subprocess.run([sys.executable, "-m", "runtime.sample",
                        "--ckpt", ckpt, "--category", cat, "--n", "4",
                        "--grid", dest, "--seed", "0"],
                       capture_output=True, text=True)
    print(f"{cat}: rc={r.returncode} {r.stdout.strip()[:120]} {r.stderr.strip()[-200:]}", flush=True)

print("gotowe:", sorted(os.listdir(OUT)), flush=True)
