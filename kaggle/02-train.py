"""
Kaggle cell 2 of 2 — training.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and Files".
Inputs: the 'gdoodle-corpus' Dataset produced by 01-prep.py, plus — on any run
after the first — the previous run's output as 'gdoodle-ckpt'.

A Kaggle session is capped at 12 hours and can die sooner without warning, so
this is built to be interrupted: everything needed to continue lands in
/kaggle/working/run every SAVE_EVERY steps, and adding that output as an input
to the next session resumes mid-stride.

One thing learned the hard way on [[g-images]]: a checkpoint cannot be pulled
out of a *running* Kaggle kernel — output only becomes downloadable once the
kernel finishes. To look at the model early, stop the kernel, fetch the output,
and resume. That costs about fifteen minutes and can save hours of training in
the wrong direction.
"""

import os
import subprocess
import sys
import glob

REPO = "https://github.com/JerzySukiennik/g-doodle.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

# ---------------------------------------------------------------- schedule --
# 32 x 4 x 512 = 65,536 tokens per step. The packed corpus is ~525M tokens, so
# one epoch is ~8,000 steps and this is two epochs.
#
# Step time is NOT estimated here on purpose. Read the s/step the first hundred
# steps print and do the arithmetic before committing a session — guessing it
# is how a run gets cut off at 90% with nothing to show.
BATCH, ACCUM, STEPS, LR_STEPS, WARMUP = 32, 4, 16000, 16000, 500

if os.path.exists(f"{WORK}/g-doodle"):
    subprocess.run(["git", "-C", f"{WORK}/g-doodle", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-doodle"], check=True)
os.chdir(f"{WORK}/g-doodle")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------- data --
corpus = None
# Recursive, because Kaggle's mount depth is not fixed: a Dataset lands at
# /kaggle/input/<slug>/, while a notebook attached through kernel_sources lands
# at /kaggle/input/notebooks/<owner>/<slug>/ — two levels deeper than the old
# pattern reached. Attaching the prep notebook directly saves pushing 800 MB
# down a home connection and back up again as a Dataset.
for candidate in sorted(glob.glob("/kaggle/input/**/meta.json", recursive=True)):
    corpus = os.path.dirname(candidate)
    break
if corpus is None:
    raise SystemExit("no corpus found — add the 'gdoodle-corpus' Dataset as an input")
print(f"corpus: {corpus}")

# ------------------------------------------------------------ resume, if any --
resume = None
for candidate in sorted(glob.glob("/kaggle/input/**/g-doodle.pt", recursive=True)):
    resume = candidate
    break
if resume:
    print(f"resuming from {resume}")
else:
    print("no checkpoint input — starting from scratch")

cmd = [
    sys.executable, "-m", "train.train",
    "--data", corpus,
    "--out", OUT,
    "--steps", str(STEPS),
    "--lr-steps", str(LR_STEPS),
    "--batch", str(BATCH),
    "--accum", str(ACCUM),
    "--warmup", str(WARMUP),
]
if resume:
    cmd += ["--resume", resume]

subprocess.run(cmd, check=True)

# A grid of samples in the output makes the checkpoint reviewable without
# downloading a gigabyte of weights first.
subprocess.run([
    sys.executable, "-m", "runtime.sample",
    "--ckpt", f"{OUT}/g-doodle.pt",
    "--grid", f"{OUT}/samples.svg",
    "--n", "1",
], check=False)

print(f"\ndone — save /kaggle/working/run as a Dataset named 'gdoodle-ckpt' "
      f"and add it as an input to the next session")
