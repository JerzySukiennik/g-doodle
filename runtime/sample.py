"""
Draw with a trained G-Doodle checkpoint and write the result as SVG.

Two modes, both the same model:

    --category dragon            the model draws from nothing
    --continue sketch.json       the model finishes a sketch you started

"Finish my sketch" needs no extra machinery: a partial drawing is just a token
prefix, so it is fed in exactly where a fresh sketch would have only BOS and the
category token. That is the whole reason this architecture was chosen over a
VAE — the interactive mode falls out of it for free.

    python -m runtime.sample --ckpt checkpoints/g-doodle.pt --grid out.svg
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import DoodleGPT, DoodleConfig
from model.tokenizer import StrokeTokenizer, strokes_to_svg, BOS, EOS


def load(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = DoodleConfig(**ckpt['config'])
    model = DoodleGPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    tokenizer = StrokeTokenizer(ckpt['categories'])
    return model, tokenizer, ckpt.get('step', 0)


def draw(model, tokenizer, category, device='cpu', prefix=None,
         temperature=0.9, top_p=0.9, max_tokens=460):
    ids = [BOS, tokenizer.category_token(category)]
    if prefix is not None:
        ids += [t for t in tokenizer.encode(prefix, category)[2:] if t != EOS]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = list(ids)
    budget = min(max_tokens, model.config.block_size - len(ids) - 1)
    for token in model.generate(idx, budget, temperature=temperature,
                                top_p=top_p, stop_token=EOS):
        out.append(token)
    return tokenizer.decode(out)


def contact_sheet(cells, cols, cell=180):
    """Lay rendered sketches out in a grid, each with its caption underneath."""
    rows = (len(cells) + cols - 1) // cols
    w, h = cols * cell, rows * (cell + 22)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#fff"/>']
    for i, (label, strokes) in enumerate(cells):
        x = (i % cols) * cell
        y = (i // cols) * (cell + 22)
        inner = strokes_to_svg(strokes, size=cell, stroke_width=2.5)
        inner = inner[inner.index('>') + 1:inner.rindex('</svg>')]
        parts.append(f'<g transform="translate({x},{y})">{inner}</g>')
        parts.append(f'<text x="{x + cell / 2}" y="{y + cell + 14}" text-anchor="middle" '
                     f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="11" '
                     f'fill="#666">{label}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--category', default=None, help='default: one of each')
    ap.add_argument('--n', type=int, default=1, help='samples per category')
    ap.add_argument('--continue-from', default=None,
                    help='JSON file with a stroke-3 array to finish')
    ap.add_argument('--out', default='sample.svg')
    ap.add_argument('--grid', default=None, help='write a contact sheet here instead')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, step = load(args.ckpt, device)
    print(f'checkpoint at step {step}, {len(tokenizer.categories)} categories')

    prefix = None
    if args.continue_from:
        with open(args.continue_from) as f:
            prefix = np.array(json.load(f), dtype=np.int32)

    categories = [args.category] if args.category else tokenizer.categories
    cells = []
    for category in categories:
        for _ in range(args.n):
            strokes = draw(model, tokenizer, category, device, prefix,
                           args.temperature, args.top_p)
            cells.append((category, strokes))
            print(f'{category:<10} {len(strokes)} strokes, '
                  f'{sum(len(s) for s in strokes)} points')

    if args.grid or len(cells) > 1:
        path = args.grid or args.out
        with open(path, 'w') as f:
            f.write(contact_sheet(cells, cols=min(5, len(cells))))
    else:
        path = args.out
        with open(path, 'w') as f:
            f.write(strokes_to_svg(cells[0][1]))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
