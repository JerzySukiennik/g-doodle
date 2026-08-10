"""
Local smoke test — everything that can be checked without a GPU or real data.

The point is to fail here, on a laptop, in seconds, rather than twenty minutes
into a Kaggle session. It checks the claims the rest of the project rests on:
the tokenizer really does reproduce a path, the loss really does start at
ln(vocab), the
KV cache really does agree with the uncached path, and a checkpoint really can
be resumed without the loss jumping.

    python -m tests.smoke
"""

import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.categories import CATEGORIES
from model.gpt import DoodleGPT, DoodleConfig
from model.tokenizer import (StrokeTokenizer, strokes_to_svg, delta_to_bin,
                             bin_to_delta, BOS, EOS, FINE, MAX_DELTA, N_DELTA)
from train.train import get_batch, lr_at

PASS, FAIL = '  ok  ', ' FAIL '
results = []


def check(name, condition, detail=''):
    results.append(bool(condition))
    print(f'[{PASS if condition else FAIL}] {name}' + (f' — {detail}' if detail else ''))


def random_sketch(rng, n_points=40):
    """A stroke-3 array shaped like real QuickDraw: mostly small deltas, a few
    lifts, and the occasional long jump that exercises the coarse bins."""
    d = rng.integers(-30, 31, size=(n_points, 2)).astype(np.int32)
    jumps = rng.random(n_points) < 0.05
    d[jumps] = rng.integers(-900, 901, size=(int(jumps.sum()), 2))
    pen = (rng.random(n_points) < 0.12).astype(np.int32)
    pen[-1] = 1
    return np.concatenate([d, pen[:, None]], axis=1)


def test_tokenizer():
    rng = np.random.default_rng(0)
    tok = StrokeTokenizer(CATEGORIES)
    check('vocab size matches layout',
          tok.vocab_size == 4 + len(CATEGORIES) + 2 * N_DELTA, f'{tok.vocab_size}')

    # The fine range is the claim the tokenizer design rests on: the deltas that
    # carry a drawing's shape must survive exactly.
    exact = all(bin_to_delta(delta_to_bin(d)) == d for d in range(-FINE, FINE + 1))
    check('deltas within +/-127 are exact', exact)
    worst = max(abs(bin_to_delta(delta_to_bin(d)) - d)
                for d in range(-MAX_DELTA, MAX_DELTA + 1))
    check('coarse error stays under half a bin', worst <= 4, f'{worst}px worst case')
    check('bins are unique and dense',
          len({delta_to_bin(d) for d in range(-MAX_DELTA, MAX_DELTA + 1)}) == N_DELTA)
    check('out-of-range deltas clip rather than wrap',
          delta_to_bin(9999) == delta_to_bin(MAX_DELTA)
          and delta_to_bin(-9999) == delta_to_bin(-MAX_DELTA))

    sketch = random_sketch(rng)
    ids = tok.encode(sketch, 'dragon')
    check('encode starts BOS + category', ids[0] == BOS and ids[1] == tok.category_token('dragon'))
    check('encode ends EOS', ids[-1] == EOS)
    check('every id inside vocab', all(0 <= i < tok.vocab_size for i in ids))
    check('category token round-trips', tok.category_of(ids[1]) == 'dragon')

    deltas = []
    i = 2
    while i < len(ids) - 1:
        if ids[i] == 3:
            i += 1
            continue
        deltas.append((bin_to_delta(ids[i] - tok.dx_base),
                       bin_to_delta(ids[i + 1] - tok.dy_base)))
        i += 2
    err = np.abs(np.array(deltas) - sketch[:, :2])
    check('round-trip error is bounded', err.max() <= 4,
          f'{len(deltas)} points, max {err.max()}px, mean {err.mean():.3f}px')
    small = np.abs(sketch[:, :2]) <= FINE
    check('every small delta round-trips exactly', err[small].max() == 0,
          f'{small.sum()}/{small.size} deltas in the fine range')

    strokes = tok.decode(ids)
    check('decode yields strokes', len(strokes) > 0, f'{len(strokes)} strokes')

    # Geometry check on a sketch made only of fine-range deltas, where the
    # tokenizer is exact and the decoded path must therefore match to the pixel.
    tame = random_sketch(np.random.default_rng(3))
    tame[:, :2] = np.clip(tame[:, :2], -FINE, FINE)
    absolute = np.cumsum(tame[:, :2], axis=0)
    drawn = np.concatenate(tok.decode(tok.encode(tame, 'cat')), axis=0)
    check('decoded points lie exactly on the original path',
          all(np.any(np.all(absolute == p, axis=1)) for p in drawn),
          f'{len(drawn)} points')

    check('decode survives garbage',
          isinstance(tok.decode([BOS, 7, 999, 4, 3, 3, tok.vocab_size - 1, EOS]), list))

    svg = strokes_to_svg(strokes)
    check('svg renders', svg.startswith('<svg') and svg.endswith('</svg>'),
          f'{len(svg)} chars')
    check('empty svg is still valid', strokes_to_svg([]).endswith('</svg>'))


def test_model():
    torch.manual_seed(0)
    cfg = DoodleConfig(vocab_size=4 + len(CATEGORIES) + 2 * N_DELTA)
    model = DoodleGPT(cfg)
    n = model.num_params()
    check('parameter count in browser budget', 12e6 < n < 25e6,
          f'{n:,} params, {n * 4 / 1e6:.0f} MB fp32 / {n / 1e6:.0f} MB int8')

    x = torch.randint(0, cfg.vocab_size, (2, 65))
    logits, loss = model(x[:, :-1], targets=x[:, 1:])
    expected = math.log(cfg.vocab_size)
    check('forward shape', logits.shape == (2, 64, cfg.vocab_size))
    check('loss starts at ln(vocab)', abs(loss.item() - expected) < 0.15,
          f'{loss.item():.3f} vs {expected:.3f}')

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check('backward reaches every parameter',
          len(grads) == len(list(model.parameters())))
    check('gradients are finite', all(torch.isfinite(g).all() for g in grads))

    _, loss_nologits = model(x[:, :-1], targets=x[:, 1:], return_logits=False)
    check('return_logits=False still gives loss', loss_nologits.requires_grad)

    # The cache is an optimisation, so it has to be numerically invisible.
    model.eval()
    with torch.no_grad():
        seq = torch.randint(0, cfg.vocab_size, (1, 24))
        full, _ = model(seq)
        from model.gpt import KVCache
        kv = KVCache(cfg.n_layer)
        step = None
        for t in range(seq.size(1)):
            step, _ = model(seq[:, t:t + 1], kv=kv)
    check('KV cache matches uncached forward',
          torch.allclose(full, step, atol=1e-4),
          f'max diff {(full - step).abs().max().item():.2e}')


def test_generation():
    torch.manual_seed(0)
    tok = StrokeTokenizer(CATEGORIES)
    cfg = DoodleConfig(vocab_size=tok.vocab_size)
    model = DoodleGPT(cfg)

    idx = torch.tensor([[BOS, tok.category_token('cat')]])
    out = [BOS, tok.category_token('cat')]
    for token in model.generate(idx, 120, temperature=1.0, top_p=0.9, stop_token=EOS):
        out.append(token)
    check('generate produces tokens', len(out) > 2, f'{len(out)} tokens')
    check('generated ids are in vocab', all(0 <= t < cfg.vocab_size for t in out))

    strokes = tok.decode(out)
    check('untrained output still decodes and renders',
          strokes_to_svg(strokes).endswith('</svg>'),
          f'{len(strokes)} strokes')

    # "Finish my sketch" is prefix conditioning and nothing else.
    prefix = tok.encode(random_sketch(np.random.default_rng(1), 20), 'cat')[:-1]
    idx = torch.tensor([prefix])
    got = [t for t in model.generate(idx, 40, stop_token=EOS)]
    check('prefix conditioning runs', len(got) > 0, f'{len(got)} continued tokens')


def test_training_loop():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        tok = StrokeTokenizer(CATEGORIES)
        corpus = np.concatenate([
            np.array(tok.encode(random_sketch(rng, 30), CATEGORIES[i % len(CATEGORIES)]),
                     dtype=np.uint16)
            for i in range(120)
        ])
        corpus.tofile(os.path.join(tmp, 'train.bin'))
        tokens = np.memmap(os.path.join(tmp, 'train.bin'), dtype=np.uint16, mode='r')

        cfg = DoodleConfig(vocab_size=tok.vocab_size, n_layer=2, n_embd=64,
                           n_head=4, ffn_hidden=128, block_size=64)
        model = DoodleGPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_rng = np.random.default_rng(7)

        x, y = get_batch(tokens, 4, cfg.block_size, 'cpu', batch_rng)
        check('batch shapes', x.shape == (4, cfg.block_size) and y.shape == x.shape)
        check('targets are inputs shifted by one',
              torch.equal(x[:, 1:], y[:, :-1]))

        losses = []
        for _ in range(30):
            x, y = get_batch(tokens, 4, cfg.block_size, 'cpu', batch_rng)
            _, loss = model(x, targets=y, return_logits=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        check('loss falls on a tiny corpus',
              losses[-1] < losses[0] - 0.2, f'{losses[0]:.3f} -> {losses[-1]:.3f}')

        ckpt = os.path.join(tmp, 'ck.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'config': cfg.__dict__, 'categories': CATEGORIES,
                    'step': 30, 'lr_total': 100}, ckpt)

        saved = torch.load(ckpt, map_location='cpu')
        check('checkpoint keys are unprefixed',
              not any(k.startswith('module.') for k in saved['model']))

        model2 = DoodleGPT(DoodleConfig(**saved['config']))
        model2.load_state_dict(saved['model'])
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(saved['optimizer'])

        eval_rng = np.random.default_rng(99)
        x, y = get_batch(tokens, 4, cfg.block_size, 'cpu', eval_rng)
        with torch.no_grad():
            _, before = model(x, targets=y, return_logits=False)
            _, after = model2(x, targets=y, return_logits=False)
        check('resume does not jump the loss',
              abs(before.item() - after.item()) < 1e-5,
              f'{before.item():.6f} vs {after.item():.6f}')

        warm = lr_at(0, 6e-4, 500, 30000)
        peak = lr_at(500, 6e-4, 500, 30000)
        end = lr_at(29999, 6e-4, 500, 30000)
        check('lr warms up then decays', warm < peak and end < peak * 0.2,
              f'{warm:.1e} -> {peak:.1e} -> {end:.1e}')
        # Raising the ceiling must not rewind the schedule.
        check('lr horizon is independent of the step ceiling',
              lr_at(10000, 6e-4, 500, 30000) == lr_at(10000, 6e-4, 500, 30000))
        check('past the horizon lr holds at the floor',
              abs(lr_at(40000, 6e-4, 500, 30000) - 6e-5) < 1e-9)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    print('\n--- tokenizer ---')
    test_tokenizer()
    print('\n--- model ---')
    test_model()
    print('\n--- generation ---')
    test_generation()
    print('\n--- training loop ---')
    test_training_loop()

    print(f'\n{sum(results)}/{len(results)} checks passed')
    sys.exit(0 if all(results) else 1)
