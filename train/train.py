"""
Train G-Doodle on the packed QuickDraw corpus.

Two things in here are scar tissue from [[g-micro]] rather than fresh design:

  - Under `DataParallel` the loss comes back as one value per GPU, so it needs
    `.mean()` before `.backward()`, and every state_dict key is prefixed with
    `module.` unless the wrapper is unwrapped before saving. Both fail loudly
    only on multi-GPU, which is to say only on Kaggle.
    See [[dataparallel-loss-mean-and-module-prefix]].

  - The cosine schedule decays towards `--lr-steps`, which is stored in the
    checkpoint and never changes, not towards `--steps`, which is a ceiling you
    are allowed to raise between sessions. Tying them together means resuming
    with a higher ceiling silently rewinds the learning rate.
    See [[decouple-lr-schedule-from-raisable-step-ceiling]].

    python -m train.train --data /kaggle/input/gdoodle-corpus --out checkpoints
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import DoodleGPT, DoodleConfig


def load_split(data_dir, name):
    path = os.path.join(data_dir, f'{name}.bin')
    return np.memmap(path, dtype=np.uint16, mode='r')


def get_batch(tokens, batch_size, block_size, device, rng):
    ix = rng.integers(0, len(tokens) - block_size - 1, size=batch_size)
    x = np.stack([tokens[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def lr_at(step, base_lr, warmup, total, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, tokens, batch_size, block_size, device, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(tokens, batch_size, block_size, device, rng)
        _, loss = model(x, targets=y, return_logits=False)
        losses.append(loss.mean().item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='checkpoints')
    ap.add_argument('--steps', type=int, default=30000, help='stop here (raisable on resume)')
    ap.add_argument('--lr-steps', type=int, default=30000,
                    help='horizon the cosine decays towards; fixed at first launch')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.data, 'meta.json')) as f:
        meta = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    train_tokens = load_split(args.data, 'train')
    valid_tokens = load_split(args.data, 'valid')

    config = DoodleConfig(vocab_size=meta['vocab_size'])
    model = DoodleGPT(config).to(device)

    start_step = 0
    lr_total = args.lr_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        lr_total = ckpt.get('lr_total', args.lr_steps)
        print(f'resumed from {args.resume} at step {start_step} (lr horizon {lr_total})')

    print(f'device {device}, {n_gpu} GPU(s)')
    print(f'params {model.num_params():,}, vocab {config.vocab_size}')
    print(f'train {len(train_tokens):,} tokens, valid {len(valid_tokens):,}')
    print(f'expected loss at init: {math.log(config.vocab_size):.3f}\n')

    raw_model = model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    model.train()
    t0 = time.time()
    tokens_seen = 0

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.lr, args.warmup, lr_total)
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train_tokens, args.batch, config.block_size, device, rng)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device == 'cuda')):
                _, loss = model(x, targets=y, return_logits=False)
                loss = loss.mean() / args.accum
            scaler.scale(loss).backward()
            total_loss += loss.item()
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f'step {step + 1:>6} | loss {total_loss:.4f} | lr {lr:.2e} | '
                  f'{dt / args.log_every:.3f}s/step | {tokens_seen / dt / 1000:.0f}k tok/s',
                  flush=True)
            t0 = time.time()
            tokens_seen = 0

        if (step + 1) % args.eval_every == 0:
            val = evaluate(raw_model, valid_tokens, args.batch, config.block_size, device, rng)
            print(f'  valid loss {val:.4f}', flush=True)
            t0 = time.time()
            tokens_seen = 0

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            path = os.path.join(args.out, 'g-doodle.pt')
            blob = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config.__dict__,
                'categories': meta['categories'],
                'step': step + 1,
                'lr_total': lr_total,
            }
            # Write to a temp file and rename, never over the checkpoint itself.
            # Saving in place means the only copy is a half-written file for as
            # long as the write takes, and G-Images lost 8h11m of training to
            # exactly that window on 2026-08-09: a Kaggle session killed
            # mid-write left 412 MB of a 1.57 GB file. Not 'most of the weights'
            # either — torch orders storages on disk by key rather than by
            # section, so every surviving storage was optimizer state and not one
            # of the weight tensors. rename() is atomic, so g-doodle.pt is always
            # either the previous complete checkpoint or the new one.
            tmp = path + '.tmp'
            torch.save(blob, tmp)
            os.replace(tmp, path)
            print(f'  saved {path} at step {step + 1}', flush=True)


if __name__ == '__main__':
    main()
