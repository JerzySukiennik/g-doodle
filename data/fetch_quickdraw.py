"""
Build the G-Doodle training corpus from Google's QuickDraw release.

Downloads the sketch-rnn `.npz` for each category, tokenises every drawing and
packs the result into one flat uint16 array plus a small JSON of metadata —
the same shape of artefact as [[g-micro]]'s `pl_train.bin`, for the same reason:
training then starts from an mmap in seconds instead of re-parsing JSON.

Run this on Kaggle, not at home. Kaggle's connection pulls ~1 GB of npz in
minutes and the packed output is saved as a Dataset every later training session
reuses; over a home uplink the same corpus would be an upload measured in hours.
See [[generate-data-where-you-will-train]].

    python -m data.fetch_quickdraw --out /kaggle/working/corpus --per-class 10000
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.categories import CATEGORIES
from model.tokenizer import StrokeTokenizer, MAX_DELTA

BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn'


def download(category, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{category}.npz')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    # 'hot dog', 'The Eiffel Tower' and friends have spaces in their object names.
    url = f'{BASE_URL}/{urllib.parse.quote(category)}.npz'
    tmp = path + '.part'
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f'{category}: download failed ({exc})') from exc
    os.replace(tmp, path)
    return path


def load_split(path, split):
    with np.load(path, encoding='latin1', allow_pickle=True) as z:
        return z[split]


def pack(split, per_class, max_points, cache_dir, tokenizer, seed, out_path):
    """Tokenise every drawing in `split` and stream it to `out_path`, shuffled.

    Shuffling matters: drawings are packed end to end and training reads random
    contiguous windows, so a window straddles two drawings. Left in category
    order, the model would learn that a cat is reliably followed by another cat
    — a correlation that exists only in the packing.

    Done in two disk passes rather than one big in-memory shuffle. At 345 x 10k
    drawings the corpus is over a gigabyte spread across three million small
    arrays, and holding all of that plus a shuffled copy is the sort of thing
    that dies twenty minutes into a session.
    """
    rng = np.random.default_rng(seed)
    dropped_long = 0
    clipped = 0
    lengths = []
    scratch = out_path + '.ordered'

    with open(scratch, 'wb') as sink:
        for category in CATEGORIES:
            path = download(category, cache_dir)
            drawings = load_split(path, split)
            if per_class:
                drawings = drawings[:per_class]

            kept = 0
            for strokes in drawings:
                strokes = np.asarray(strokes, dtype=np.int32)
                if len(strokes) > max_points:
                    dropped_long += 1
                    continue
                # Deltas past MAX_DELTA are clipped by the tokenizer. Measured on
                # the real release the maximum is 1728 against a ceiling of 1791,
                # so this should stay at zero — a nonzero count means the range
                # assumption is wrong again and the bins need widening.
                clipped += int(np.sum(np.abs(strokes[:, :2]) > MAX_DELTA))
                ids = np.array(tokenizer.encode(strokes, category), dtype=np.uint16)
                ids.tofile(sink)
                lengths.append(ids.size)
                kept += 1

            print(f'  {category:<24} {kept:>7,} drawings  '
                  f'{sum(lengths) / 1e6:>8.1f}M tokens', flush=True)

    # Second pass: rewrite in a random drawing order. Only the offsets live in
    # memory (8 bytes per drawing), and the gather reads through an mmap, so the
    # corpus is never held whole.
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    ordered = np.memmap(scratch, dtype=np.uint16, mode='r')
    order = rng.permutation(len(lengths))

    print(f'  shuffling {len(lengths):,} drawings...', flush=True)
    with open(out_path, 'wb') as sink:
        buf = []
        for i in order:
            buf.append(np.asarray(ordered[offsets[i]:offsets[i + 1]]))
            if len(buf) >= 20000:
                np.concatenate(buf).tofile(sink)
                buf = []
        if buf:
            np.concatenate(buf).tofile(sink)

    del ordered
    os.remove(scratch)
    return int(lengths.sum()), dropped_long, clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--cache', default=None, help='where to keep the raw npz (default: <out>/raw)')
    ap.add_argument('--per-class', type=int, default=10000,
                    help='drawings per category from the train split (0 = all 70k). '
                         '345 x 10k is ~525M tokens, about 1 GB packed')
    ap.add_argument('--valid-per-class', type=int, default=500)
    ap.add_argument('--max-points', type=int, default=200,
                    help='drop sketches longer than this; 200 points = 400+ tokens')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.cache or os.path.join(args.out, 'raw')
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = StrokeTokenizer(CATEGORIES)
    print(f'{len(CATEGORIES)} categories, vocab {tokenizer.vocab_size}\n')

    print('train split:')
    train, dropped, clipped = pack('train', args.per_class, args.max_points,
                                   cache_dir, tokenizer, args.seed,
                                   os.path.join(args.out, 'train.bin'))
    print('\nvalid split:')
    valid, _, _ = pack('valid', args.valid_per_class, args.max_points,
                       cache_dir, tokenizer, args.seed + 1,
                       os.path.join(args.out, 'valid.bin'))

    meta = {
        'categories': CATEGORIES,
        'vocab_size': tokenizer.vocab_size,
        'train_tokens': int(train),
        'valid_tokens': int(valid),
        'max_points': args.max_points,
        'per_class': args.per_class,
        'dropped_too_long': int(dropped),
        'deltas_clipped': int(clipped),
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\ntrain {train:,} tokens ({train * 2 / 1e6:.0f} MB)')
    print(f'valid {valid:,} tokens')
    print(f'dropped as too long: {dropped:,}')
    print(f'deltas clipped: {clipped:,}  (expected 0 — anything else is a bug)')


if __name__ == '__main__':
    main()
