# G-Doodle

A transformer that draws. You type what you want, it draws it stroke by stroke —
live, in the browser, with no server.

Third model of the G family, after [G-Micro](https://github.com/JerzySukiennik/g-micro)
(a 117M Polish language model trained from scratch) and G-Images (a diffusion
image editor trained from scratch). Same principle as both: everything trained
from zero on free hardware.

```
"narysuj mi smoka"  ->  dragon  ->  BOS CAT (dx dy)* PEN_UP (dx dy)* ... EOS
```

## How it works

It is a decoder-only transformer, and it has no idea it is drawing. It predicts
the next token in a sequence, exactly as a language model does — the tokens just
happen to be pen movements instead of words. That equivalence is the point: the
architecture, training loop and multi-GPU handling are lifted straight from
G-Micro.

- **~18.4M parameters**, 10 layers, d=384, RoPE + RMSNorm + SwiGLU
- **345 categories** — everything in Google's QuickDraw release
- **~525M training tokens** from 3.45M human drawings
- trained on **Kaggle's free T4 x2**, like every other G model

## What it can and cannot draw

345 things, and that boundary is hard. Cat, dragon, octopus, castle, pizza,
piano, the Eiffel Tower, penguin, mermaid, submarine — yes. Anything outside the
list — no, and it says so rather than drawing the nearest thing and letting you
assume it tried.

Notably **QuickDraw contains no `rocket`**, which was the first thing this
project wanted to draw.

Free text is *resolved*, not understood: `runtime/prompt.py` matches Polish and
English against an alias table of 766 entries. "narysuj mi smoka" finds `dragon`
because the string contains "smok".

## Two modes

Both come from the same model, with no extra machinery:

- **draw from nothing** — feed `BOS, <category>` and sample
- **finish my sketch** — your strokes are just a token prefix

## Layout

```
model/tokenizer.py     strokes <-> tokens (non-uniform bins, see SPEC)
model/gpt.py           the transformer
data/categories.py     the 345 classes
data/fetch_quickdraw.py  build the corpus (run on Kaggle)
train/train.py         training loop
runtime/prompt.py      free text -> category
runtime/sample.py      draw, render to SVG
kaggle/                the two notebook cells
tests/smoke.py         35 checks that run on a laptop in seconds
```

## Running it

```bash
python -m tests.smoke                                    # verify everything first
python -m data.fetch_quickdraw --out corpus              # ~5 GB down, ~1 GB out
python -m train.train --data corpus --out checkpoints
python -m runtime.sample --ckpt checkpoints/g-doodle.pt --grid out.svg
```

On Kaggle, run `kaggle/01-prep.py` then `kaggle/02-train.py` instead.

Full design notes, including the measurement that broke the first tokenizer, are
in [SPEC.md](SPEC.md).

## Data

[QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) by Google
Creative Lab, CC BY 4.0. 50 million drawings by people who had twenty seconds
and a mouse.

## Licence

MIT.
