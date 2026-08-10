"""
Kaggle cell 1 of 2 — build the token binaries.

Run this ONCE. It pulls all 345 QuickDraw sketch-rnn archives, tokenises every
drawing and packs them into train.bin / valid.bin under /kaggle/working/corpus,
which you then save as a Dataset and feed to the training notebook. Doing it
here rather than uploading from home is deliberate: Kaggle's connection pulls
~5 GB of npz in minutes, a domestic upload of the 1 GB result would not.

Settings: GPU OFF (this stage is CPU only), Internet ON.
Expect roughly 30-60 minutes, most of it download.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-doodle.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/corpus"

PER_CLASS = 10000        # 345 x 10k drawings -> ~525M tokens, ~1 GB packed
VALID_PER_CLASS = 500

# A stale checkout from an earlier attempt in the same session would silently
# run old code — pull rather than assume the clone is fresh.
if os.path.exists(f"{WORK}/g-doodle"):
    subprocess.run(["git", "-C", f"{WORK}/g-doodle", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-doodle"], check=True)
os.chdir(f"{WORK}/g-doodle")

subprocess.run([
    sys.executable, "-m", "data.fetch_quickdraw",
    "--out", OUT,
    "--per-class", str(PER_CLASS),
    "--valid-per-class", str(VALID_PER_CLASS),
], check=True)

# The raw npz archives are ~5 GB and are only an intermediate — keeping them
# would make the saved Dataset six times larger for no benefit, and every
# training session would then wait on them.
raw = f"{OUT}/raw"
if os.path.isdir(raw):
    size = sum(os.path.getsize(os.path.join(raw, f)) for f in os.listdir(raw))
    subprocess.run(["rm", "-rf", raw], check=True)
    print(f"removed {size / 1e9:.1f} GB of raw npz")

for name in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, name)
    print(f"{name:<14} {os.path.getsize(path) / 1e6:>9.1f} MB")

print("\nSave /kaggle/working/corpus as a Dataset named 'gdoodle-corpus', "
      "then run 02-train.py with it as an input.")
