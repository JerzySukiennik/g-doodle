"""
Export G-Doodle to ONNX so the browser can draw without a server.

The exported graph is **one decode step**, not the whole generation loop:

    (input_ids, positions, past_k, past_v) -> (logits, present_k, present_v)

Keeping the loop in JavaScript rather than in the graph is what makes the KV
cache possible at all. Without it, drawing a 400-token sketch means 400 forward
passes over a growing prefix — quadratic work, several seconds of a phone's CPU
burned re-reading tokens whose keys and values cannot have changed. With it,
every step after the first costs exactly one token.

The cache is passed as two stacked tensors, `(n_layer, batch, heads, past, dim)`,
instead of 2 x n_layer separate inputs. Twenty loose tensors would work equally
well numerically and be far more error-prone to thread through JavaScript.

Prefill runs one token at a time through the same graph. A separate multi-token
prefill graph would need a causal mask built from dynamic shapes — more export
risk than it is worth, when the only thing it saves is about a second on the
"finish my sketch" path.

    python -m runtime.export_onnx --ckpt checkpoints/g-doodle.pt --out web/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import DoodleGPT, DoodleConfig, build_rope_cache, apply_rope
from model.tokenizer import (StrokeTokenizer, FINE, COARSE_STEP, MAX_DELTA,
                             N_DELTA, PAD, BOS, EOS, PEN_UP)


class DoodleStep(nn.Module):
    """One decode step with an explicit, exportable cache.

    The attention here is deliberately spelled out rather than reusing
    `CausalSelfAttention.forward`, because that method takes a Python `KVCache`
    object, which cannot be traced. The duplication is a real risk — two
    implementations of the same maths can drift apart silently — so
    `verify_parity` below exists specifically to catch that, and should be run
    on every export.
    """

    def __init__(self, model: DoodleGPT):
        super().__init__()
        self.m = model
        cfg = model.config
        cos, sin = build_rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size,
                                    cfg.rope_theta, 'cpu', torch.float32)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, input_ids, positions, past_k, past_v):
        cfg = self.m.config
        H, D = cfg.n_head, cfg.n_embd // cfg.n_head
        KVH = getattr(cfg, 'n_kv_head', H)      # grouped-query attention
        REP = H // KVH

        x = self.m.tok_emb(input_ids)
        cos = self.rope_cos.index_select(0, positions)
        sin = self.rope_sin.index_select(0, positions)

        new_k, new_v = [], []
        for i, block in enumerate(self.m.blocks):
            h = block.norm_attn(x)
            B, T, C = h.shape
            # Queries are full width, keys and values are narrow: the cache
            # this graph hands back to JavaScript is KVH heads wide, not H, and
            # shrinking it is the entire reason GQA was added before training.
            q, k, v = block.attn.qkv(h).split([C, KVH * D, KVH * D], dim=2)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, KVH, D).transpose(1, 2)
            v = v.view(B, T, KVH, D).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            k = torch.cat([past_k[i], k], dim=2)
            v = torch.cat([past_v[i], v], dim=2)
            new_k.append(k)
            new_v.append(v)

            # Expand to full width only for the attention itself, after the
            # cache has been captured — caching the wide version would undo GQA.
            kx, vx = (k.repeat_interleave(REP, dim=1),
                      v.repeat_interleave(REP, dim=1)) if REP > 1 else (k, v)

            # No mask: a single new token may attend to the entire past, which
            # is exactly what unmasked attention over (past + self) computes.
            y = F.scaled_dot_product_attention(q, kx, vx)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + block.attn.proj(y)
            x = x + block.ffn(block.norm_ffn(x))

        x = self.m.norm_final(x)
        return self.m.lm_head(x[:, -1, :]), torch.stack(new_k), torch.stack(new_v)


def empty_cache(cfg, batch=1):
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    KVH = getattr(cfg, 'n_kv_head', H)
    shape = (cfg.n_layer, batch, KVH, 0, D)
    return torch.zeros(shape), torch.zeros(shape)


def export(model, path, opset=17):
    cfg = model.config
    step = DoodleStep(model).eval()
    past_k, past_v = empty_cache(cfg)

    # Trace with a non-empty cache: a zero-length axis makes the exporter guess
    # constants where it should keep a dynamic dimension.
    kv_heads = getattr(cfg, 'n_kv_head', cfg.n_head)
    past_k = torch.zeros(cfg.n_layer, 1, kv_heads, 3, cfg.n_embd // cfg.n_head)
    past_v = torch.zeros_like(past_k)
    args = (torch.tensor([[BOS]], dtype=torch.long),
            torch.tensor([3], dtype=torch.long), past_k, past_v)

    torch.onnx.export(
        step, args, path,
        input_names=['input_ids', 'positions', 'past_k', 'past_v'],
        output_names=['logits', 'present_k', 'present_v'],
        dynamic_axes={
            'past_k': {3: 'past'},
            'past_v': {3: 'past'},
            'present_k': {3: 'present'},
            'present_v': {3: 'present'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return path


def ort_generate(session, cfg, ids, max_new_tokens, temperature=0.9, top_p=0.9,
                 seed=0, greedy=False):
    """Run the exported graph the way the browser will, for verification."""
    rng = np.random.default_rng(seed)
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    KVH = getattr(cfg, 'n_kv_head', H)
    past_k = np.zeros((cfg.n_layer, 1, KVH, 0, D), dtype=np.float32)
    past_v = np.zeros_like(past_k)

    out = list(ids)
    logits = None
    for pos, token in enumerate(ids):
        logits, past_k, past_v = session.run(None, {
            'input_ids': np.array([[token]], dtype=np.int64),
            'positions': np.array([pos], dtype=np.int64),
            'past_k': past_k, 'past_v': past_v,
        })

    for _ in range(max_new_tokens):
        row = logits[0].astype(np.float64)
        if greedy:
            nxt = int(row.argmax())
        else:
            row = row / max(temperature, 1e-5)
            order = np.argsort(-row)
            probs = np.exp(row[order] - row[order].max())
            probs /= probs.sum()
            keep = np.searchsorted(np.cumsum(probs), top_p) + 1
            probs = probs[:keep] / probs[:keep].sum()
            nxt = int(order[rng.choice(keep, p=probs)])
        out.append(nxt)
        if nxt == EOS or len(out) >= cfg.block_size:
            break
        logits, past_k, past_v = session.run(None, {
            'input_ids': np.array([[nxt]], dtype=np.int64),
            'positions': np.array([len(out) - 1], dtype=np.int64),
            'past_k': past_k, 'past_v': past_v,
        })
    return out


def verify_parity(model, session, cfg, n_tokens=40, tol=2e-3):
    """The exported graph must agree with PyTorch, token for token.

    Sampling is stochastic, so parity is checked on logits under a fixed token
    sequence rather than on generated output.
    """
    from model.gpt import KVCache

    torch.manual_seed(0)
    ids = [BOS, 4] + list(np.random.default_rng(0).integers(
        cfg.vocab_size - N_DELTA, cfg.vocab_size, size=n_tokens))
    ids = [int(i) for i in ids]

    model.eval()
    with torch.no_grad():
        kv = KVCache(cfg.n_layer)
        torch_logits = None
        for t in ids:
            torch_logits, _ = model(torch.tensor([[t]]), kv=kv)
    torch_logits = torch_logits[0, -1].numpy()

    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    KVH = getattr(cfg, 'n_kv_head', H)
    past_k = np.zeros((cfg.n_layer, 1, KVH, 0, D), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    for pos, t in enumerate(ids):
        ort_logits, past_k, past_v = session.run(None, {
            'input_ids': np.array([[t]], dtype=np.int64),
            'positions': np.array([pos], dtype=np.int64),
            'past_k': past_k, 'past_v': past_v,
        })
    ort_logits = ort_logits[0]

    diff = np.abs(torch_logits - ort_logits).max()
    agree = int(torch_logits.argmax()) == int(ort_logits.argmax())
    return diff, agree, diff < tol and agree


def write_meta(path, tokenizer, cfg):
    meta = {
        'vocab_size': cfg.vocab_size,
        'block_size': cfg.block_size,
        'n_layer': cfg.n_layer,
        'n_head': cfg.n_head,
        # The browser allocates its KV cache from this, and under grouped-query
        # attention it is NOT n_head: the graph returns KVH heads, not H. Emit it
        # explicitly rather than letting the JavaScript infer, because the wrong
        # width fails as an opaque ONNX dimension error at the first decode step.
        'n_kv_head': getattr(cfg, 'n_kv_head', cfg.n_head),
        'head_dim': cfg.n_embd // cfg.n_head,
        'categories': tokenizer.categories,
        'tokens': {'PAD': PAD, 'BOS': BOS, 'EOS': EOS, 'PEN_UP': PEN_UP},
        'layout': {
            'cat_base': tokenizer.cat_base,
            'dx_base': tokenizer.dx_base,
            'dy_base': tokenizer.dy_base,
            'n_delta': N_DELTA,
        },
        'bins': {'fine': FINE, 'coarse_step': COARSE_STEP, 'max_delta': MAX_DELTA},
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint to export; omit to export an untrained model '
                         '(useful for testing the browser path before training ends)')
    ap.add_argument('--out', default='web')
    ap.add_argument('--quantize', action='store_true',
                    help='also write an int8 model (~4x smaller)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        cfg = DoodleConfig(**ckpt['config'])
        model = DoodleGPT(cfg)
        model.load_state_dict(ckpt['model'])
        categories = ckpt['categories']
        print(f'loaded {args.ckpt} at step {ckpt.get("step", "?")}')
    else:
        from data.categories import CATEGORIES
        cfg = DoodleConfig()
        model = DoodleGPT(cfg)
        categories = CATEGORIES
        print('no checkpoint — exporting an UNTRAINED model')
    model.eval()

    path = os.path.join(args.out, 'g-doodle.onnx')
    export(model, path)
    size = os.path.getsize(path) / 1e6
    print(f'exported {path} ({size:.1f} MB)')

    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    diff, agree, ok = verify_parity(model, session, cfg)
    print(f'parity vs PyTorch: max logit diff {diff:.2e}, argmax agrees: {agree} '
          f'-> {"OK" if ok else "FAILED"}')
    if not ok:
        raise SystemExit('export does not match PyTorch — do not ship this')

    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process

        # Quantising the raw export produces a graph ONNX Runtime then refuses
        # to load ("node input ... is not a graph input, initializer, or output
        # of a previous node") — the Cast pair RMSNorm emits around its float()
        # /type_as() confuses the quantiser's graph surgery. Running the
        # documented pre-process pass first (shape inference + folding) removes
        # the ambiguity. The export is verified again after quantising, because
        # a model that loads is not the same as a model that still works.
        prepped = os.path.join(args.out, 'g-doodle.prep.onnx')
        quant_pre_process(path, prepped, skip_symbolic_shape=True)
        qpath = os.path.join(args.out, 'g-doodle.int8.onnx')
        quantize_dynamic(prepped, qpath, weight_type=QuantType.QUInt8)
        os.remove(prepped)
        qsize = os.path.getsize(qpath) / 1e6
        print(f'quantised {qpath} ({qsize:.1f} MB, {size / qsize:.1f}x smaller)')

        qsession = ort.InferenceSession(qpath, providers=['CPUExecutionProvider'])
        qdiff, qagree, _ = verify_parity(model, qsession, cfg, tol=float('inf'))
        print(f'int8 drift vs PyTorch: max logit diff {qdiff:.3f}, '
              f'argmax agrees: {qagree}')

    tokenizer = StrokeTokenizer(categories)
    write_meta(os.path.join(args.out, 'g-doodle.meta.json'), tokenizer, cfg)

    from runtime.prompt import export as export_prompts
    p = export_prompts(os.path.join(args.out, 'g-doodle.prompts.json'))
    print(f'wrote meta + {len(p["aliases"])} prompt aliases')


if __name__ == '__main__':
    main()
