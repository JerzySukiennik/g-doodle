# Running G-Doodle on Kaggle

Two cells, same pattern as G-Micro and G-Images.

## 1. `01-prep.py` — build the corpus

- Accelerator: **None** (CPU only)
- Internet: **ON**
- Expect 30–60 min, most of it downloading ~5 GB of QuickDraw archives

Output lands in `/kaggle/working/corpus` (~1 GB). Save it as a Dataset named
**`gdoodle-corpus`**. The raw npz files are deleted before saving — they are an
intermediate and would make the Dataset six times larger.

## 2. `02-train.py` — train

- Accelerator: **GPU T4 x2**
- Internet: **ON**
- Persistence: **Variables and Files**
- Inputs: `gdoodle-corpus`, plus `gdoodle-ckpt` on any run after the first

Read the `s/step` from the first hundred steps and multiply before committing —
a Kaggle session is capped at 12 hours and a run cut off at 90% leaves nothing.

Checkpoints land in `/kaggle/working/run` every 1000 steps. Save that output as
**`gdoodle-ckpt`** and add it as an input next session to resume mid-stride.

## Looking at the model early

Output is only downloadable once a kernel **finishes** — there is no way to pull
a checkpoint out of a running session. To review early: stop the kernel, fetch
the output, resume from it. Costs ~15 minutes, saves hours of training in the
wrong direction. (Learned on G-Images.)

## Sanity checks

- loss at step 0 should be ≈ **7.43** (`ln(1691)`); much lower means data is
  leaking into the targets
- `deltas clipped` in the prep output should be **0**
- the 30 h/week GPU quota is shared across every project — check `kernels status`
  before starting
