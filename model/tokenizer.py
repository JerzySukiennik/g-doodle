"""
G-Doodle tokenizer — turns a drawing into a sequence of integers and back.

A QuickDraw sketch arrives in "stroke-3" format: rows of (dx, dy, pen), where
dx/dy is the move from the previous point and pen=1 marks the last point of a
stroke. Every point becomes two tokens, one per axis, plus a separator per
stroke.

## Why the bins are shaped the way they are

The first version of this file assumed deltas fit in [-255, 255], because that
is the range QuickDraw's *simplified* drawings live in. Measured against the
actual sketch-rnn `.npz` release, that was wrong: deltas reach +/-1728, and while
only ~0.4-1.2% of individual deltas exceed 255, **26-52% of drawings contain at
least one** (measured across cat/house/guitar). Clipping would therefore not
damage a rare drawing, it would damage a third of them, and the damage lands
disproportionately on the long pen-up jumps between strokes (75-94% of the big
deltas start a new stroke) — which is to say it would scramble layout, the one
thing a viewer notices instantly.

So the bins are non-uniform, matched to that measured distribution:

  - |d| <= 127  ->  exact, one bin per integer. This is 94-98% of all deltas,
                    and it is *lossless* — the part of a drawing that carries
                    its shape is reproduced perfectly.
  - |d| >  127  ->  bins 8 units wide out to +/-1791, decoded at the midpoint.
                    Worst-case error 4px on a jump longer than 127px, i.e. under
                    3% — invisible, and almost always on an invisible pen-up move.

Token layout:

    0                PAD      (loss is masked on it)
    1                BOS
    2                EOS
    3                PEN_UP   (end of stroke; the next pair is the jump)
    4 .. 4+C-1       one token per category
    then             671 dx tokens, then 671 dy tokens

A drawing reads: BOS, CAT, (dx dy)*, PEN_UP, (dx dy)*, PEN_UP, ..., EOS
"""

import numpy as np

FINE = 127            # |d| <= FINE is stored exactly
COARSE_STEP = 8       # bin width above FINE
MAX_DELTA = 1791      # observed maximum is 1728; anything past this is clipped

N_COARSE = (MAX_DELTA - FINE + COARSE_STEP - 1) // COARSE_STEP     # 208 per side
N_DELTA = (2 * FINE + 1) + 2 * N_COARSE                            # 671 per axis

PAD, BOS, EOS, PEN_UP = 0, 1, 2, 3
N_SPECIAL = 4


def delta_to_bin(d: int) -> int:
    d = int(np.clip(d, -MAX_DELTA, MAX_DELTA))
    if abs(d) <= FINE:
        return d + FINE
    k = (abs(d) - FINE + COARSE_STEP - 1) // COARSE_STEP           # 1 .. N_COARSE
    base = 2 * FINE + 1
    return base + (k - 1) + (0 if d > 0 else N_COARSE)


def bin_to_delta(b: int) -> int:
    base = 2 * FINE + 1
    if b < base:
        return b - FINE
    b -= base
    negative = b >= N_COARSE
    k = (b - N_COARSE if negative else b) + 1
    magnitude = FINE + k * COARSE_STEP - COARSE_STEP // 2          # bin midpoint
    return -magnitude if negative else magnitude


class StrokeTokenizer:

    def __init__(self, categories):
        self.categories = list(categories)
        self.cat_base = N_SPECIAL
        self.dx_base = self.cat_base + len(self.categories)
        self.dy_base = self.dx_base + N_DELTA
        self.vocab_size = self.dy_base + N_DELTA
        self._cat_index = {name: i for i, name in enumerate(self.categories)}

    # -- ids -----------------------------------------------------------------

    def category_token(self, name) -> int:
        return self.cat_base + self._cat_index[name]

    def category_of(self, token: int):
        i = token - self.cat_base
        return self.categories[i] if 0 <= i < len(self.categories) else None

    # -- encode --------------------------------------------------------------

    def encode(self, strokes, category) -> list:
        """strokes: (N, 3) int array in stroke-3 format. Returns a token list."""
        out = [BOS, self.category_token(category)]
        for dx, dy, pen in np.asarray(strokes, dtype=np.int32):
            out.append(self.dx_base + delta_to_bin(dx))
            out.append(self.dy_base + delta_to_bin(dy))
            if pen:
                out.append(PEN_UP)
        if out[-1] == PEN_UP:
            out.pop()                 # a trailing lift carries no information
        out.append(EOS)
        return out

    # -- decode --------------------------------------------------------------

    def decode(self, tokens):
        """Returns a list of strokes, each an (M, 2) float array of absolute points.

        Malformed streams are tolerated rather than raised on: a sampled model
        will occasionally emit a dy where a dx was expected, and a generator that
        crashes on its own output is useless in a live demo. Stray tokens are
        skipped.
        """
        strokes, current = [], []
        x = y = 0.0
        pen_down = False
        pending_dx = None

        for t in tokens:
            if t in (BOS, PAD):
                continue
            if t == EOS:
                break
            if t == PEN_UP:
                if len(current) > 1:
                    strokes.append(np.array(current, dtype=np.float32))
                current = []
                pen_down = False
                pending_dx = None
                continue
            if t < self.dx_base:
                continue                                     # category token
            if t < self.dy_base:
                pending_dx = bin_to_delta(t - self.dx_base)
                continue
            if pending_dx is None:
                continue                                     # dy without its dx
            x += pending_dx
            y += bin_to_delta(t - self.dy_base)
            pending_dx = None
            if pen_down:
                current.append((x, y))
            else:
                current = [(x, y)]
                pen_down = True

        if len(current) > 1:
            strokes.append(np.array(current, dtype=np.float32))
        return strokes


def strokes_to_svg(strokes, size=256, stroke_width=3.0, colour='#111'):
    """Render decoded strokes as a standalone SVG, scaled to fit `size`."""
    if not strokes:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"></svg>'

    pts = np.concatenate(strokes, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    pad = size * 0.08
    scale = (size - 2 * pad) / span

    paths = []
    for s in strokes:
        q = (s - lo) * scale + pad
        d = 'M ' + ' L '.join(f'{px:.1f},{py:.1f}' for px, py in q)
        paths.append(f'<path d="{d}"/>')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<g fill="none" stroke="{colour}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        + ''.join(paths) +
        '</g></svg>'
    )


if __name__ == '__main__':
    print(f'delta bins per axis: {N_DELTA} '
          f'(exact within +/-{FINE}, step {COARSE_STEP} out to +/-{MAX_DELTA})')
    worst = max(abs(bin_to_delta(delta_to_bin(d)) - d) for d in range(-MAX_DELTA, MAX_DELTA + 1))
    exact = sum(bin_to_delta(delta_to_bin(d)) == d for d in range(-FINE, FINE + 1))
    print(f'exact in fine range: {exact}/{2 * FINE + 1}, worst error overall: {worst}px')
