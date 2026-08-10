# G-Doodle — specification

A transformer that draws. You type what you want in Polish, it draws it stroke by
stroke, live, in your browser. Third model of the G family after [[g-micro]] and
[[g-images]], and the first one that runs entirely on the visitor's machine.

---

## 1. What it is

G-Doodle is a decoder-only transformer over pen strokes. It has no idea it is
drawing: it predicts the next token in a sequence, exactly as G-Micro predicts
the next token of Polish text. Only the alphabet differs — instead of words, the
tokens are pen movements.

That equivalence is the whole reason this project is cheap. The architecture,
the training loop, the checkpoint format, the DataParallel workarounds and the
multi-GPU scars all transfer unchanged from G-Micro. What had to be designed from
scratch is the tokenizer and the text-to-category layer, and nothing else.

**Size:** ~18.4M parameters, 10 layers, d=384. 73 MB in fp32, ~18 MB quantised to
int8 — the number that matters, because the model is downloaded by anyone who
opens the page.

## 2. What it can and cannot do

It draws **345 things** — every category in Google's QuickDraw release. That is
the model's entire world and the boundary is hard:

- **Yes:** cat, dragon, octopus, castle, pizza, piano, the Eiffel Tower,
  hamburger, penguin, mermaid, tornado, submarine...
- **No:** anything outside the 345. Most notably **there is no rocket** —
  it was the obvious first pick for this project and QuickDraw simply does not
  contain it.

Free text is resolved against an alias table (`runtime/prompt.py`), not
understood. "narysuj mi smoka" finds `dragon` because the string contains
"smok", not because anything in the system knows Polish. When the text matches
nothing, the UI says so and offers the nearest categories that do exist — it
does not quietly draw a helicopter and let you assume it tried to draw a rocket.

This limit is not a shortcut, it is what free data buys. Open-vocabulary drawing
is the same class of problem as open-domain video generation: it needs a corpus
that does not exist for free.

## 3. Two ways to use it

Both fall out of the same model with no extra machinery:

1. **Draw from nothing.** Feed `BOS, <category>` and sample to `EOS`.
2. **Finish my sketch.** A partial drawing is just a token prefix, so the person's
   strokes are tokenised and fed where the empty prefix would have gone. This is
   the interaction that makes the thing feel alive, and it costs nothing.

Choosing a decoder-only transformer over the original SketchRNN VAE was a trade:
prefix conditioning and full code reuse from G-Micro, against losing a latent
space and therefore losing morphing between two drawings. If morphing is ever
wanted, it needs a different architecture, not a bigger checkpoint.

## 4. Tokenizer

QuickDraw's sketch-rnn release stores each drawing as `(dx, dy, pen)` rows. Each
point becomes two tokens — one per axis — plus a separator per stroke:

```
BOS, CAT, (dx dy)*, PEN_UP, (dx dy)*, PEN_UP, ..., EOS
```

Vocabulary: 4 special + 345 categories + 2 x 671 delta bins = **1691**.

### Bin design, and the assumption that was wrong

The first version assumed deltas fit in ±255, because that is the range
QuickDraw's *simplified* drawings live in. Measured against the actual release,
that was false: deltas reach **±1728**. Only 0.4–1.2% of individual deltas exceed
255 — but **26–52% of drawings contain at least one**, and 75–94% of those big
deltas are the pen-up jump that starts a new stroke. Clipping would therefore not
have damaged a rare drawing; it would have scrambled the layout of a third of
them.

So the bins are non-uniform and matched to the measured distribution:

| range | bins | error |
|---|---|---|
| \|d\| ≤ 127 | one per integer, 255 bins | **exact** — 94–98% of all deltas |
| \|d\| > 127 | width 8, out to ±1791, 416 bins | ≤ 4px, i.e. under 3% |

The part of a drawing that carries its shape is reproduced perfectly; the error
lives only on long jumps, which are almost always invisible pen-up moves.

## 5. Data

`data/fetch_quickdraw.py`, run **on Kaggle** ([[generate-data-where-you-will-train]]):
downloads all 345 archives (~5 GB), tokenises, and packs to a flat uint16 array.

- 10,000 drawings per category → 3.45M drawings, **~525M tokens**, ~1 GB
- validation: 500 per category
- sketches over 200 points are dropped (measured: essentially none — p99 is 125)

Drawings are shuffled across categories before packing. Training reads random
contiguous windows, so a window straddles two drawings; left in category order
the model would learn that a cat is reliably followed by another cat, a
correlation that exists only in the packing. The shuffle is a two-pass disk
shuffle, because the corpus does not fit in memory twice.

## 6. Training

- Kaggle T4×2, `DataParallel` (`loss.mean()`, unwrap before saving —
  [[dataparallel-loss-mean-and-module-prefix]])
- batch 32 × accum 4 × block 512 = 65,536 tokens/step
- ~8,000 steps per epoch; two epochs planned
- AdamW, lr 6e-4 cosine to 6e-5, warmup 500, grad clip 1.0, fp16 autocast
- the cosine decays towards `--lr-steps`, stored in the checkpoint, *not* towards
  the raisable `--steps` ceiling ([[decouple-lr-schedule-from-raisable-step-ceiling]])

Loss at initialisation must be ≈ ln(1691) = 7.43. Anything much lower means data
is leaking into the targets.

**Step time is deliberately not predicted here.** Read it off the first hundred
steps and do the arithmetic before committing a 12-hour session.

## 7. Serving

Target: **ONNX in the browser**, int8, via onnxruntime-web, with the KV cache
exposed as graph inputs/outputs so each token costs one step instead of a full
re-read of the prefix. At ~18M parameters that should be a few milliseconds per
token on WASM — a 400-token drawing appearing over a couple of seconds, which is
the right speed anyway because watching it draw *is* the feature.

This is the first G model that does not need the Mac at home to be awake. That
matters for [[narew-labs]], where Chat and Image Studio both go dark when it is
off.

## 8. Open questions

- int8 quantisation quality has not been measured yet; if it degrades strokes,
  fp16 at 37 MB is the fallback.
- Whether 10k drawings per category is the right point on the
  quality-versus-download-size curve — 70k are available.
- A "guess what I'm drawing" mode would need a second, tiny classifier (~2h GPU).
  Not in scope for v1.
