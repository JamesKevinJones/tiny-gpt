"""Train the GPT on data/train.bin.

    uv run train.py                      full run (~2 h on an RTX 4050 at ~38k tok/s)
    uv run train.py --max_iters 50       quick smoke test
    uv run train.py --resume             continue from out/ckpt.pt after a stop

Ctrl+C is safe: the latest checkpoint is written at every eval.
"""
import argparse
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

p = argparse.ArgumentParser()
p.add_argument("--max_iters", type=int, default=8000)
p.add_argument("--batch_size", type=int, default=32)  # sequences per micro-batch
p.add_argument("--grad_accum", type=int, default=2)  # micro-batches per optimizer step
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--eval_interval", type=int, default=500)
p.add_argument("--eval_batches", type=int, default=40)
p.add_argument("--resume", action="store_true")
p.add_argument("--out", default=os.path.join(ROOT, "out"))
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# bf16 autocast: matmuls run in bfloat16 (Ada GPUs have fast bf16 tensor cores),
# weights and optimizer state stay float32. No loss scaling needed, unlike fp16.
autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
ckpt_path = os.path.join(args.out, "ckpt.pt")
if args.resume:
    # Load to CPU: on the GPU this 330 MB copy (weights + optimizer state) would sit
    # beside the live model and push a 6 GB card into slow shared memory.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
else:
    cfg = GPTConfig(vocab_size=tok.vocab_size)
model = GPT(cfg).to(device)
opt = model.configure_optimizer(args.lr, weight_decay=0.1, device_type=device)
start_iter, best_val = 0, float("inf")
if args.resume:
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])
    start_iter, best_val = ckpt["iter"], ckpt["best_val"]
    del ckpt
print(f"model: {model.num_params() / 1e6:.1f}M params | {asdict(cfg)}")


def get_batch(split):
    # Re-open the memmap every call: a long-lived np.memmap slowly leaks memory.
    data = np.memmap(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(it):
    # Linear warmup, then cosine decay down to 10% of the peak.
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = min(1.0, (it - args.warmup) / max(1, args.max_iters - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_batches)
        for k in range(args.eval_batches):
            with autocast:
                losses[k] = model(*get_batch(split))[1].item()
        out[split] = losses.mean().item()
    prompt = torch.tensor([tok.encode("Once upon a time")], device=device)
    with autocast:
        sample = tok.decode(model.generate(prompt, 120, stop_token=tok.eot)[0].tolist())
    model.train()
    return out, sample


tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
print(f"{tokens_per_step:,} tokens/step, {args.max_iters - start_iter:,} steps to go")
os.makedirs(args.out, exist_ok=True)
x, y = get_batch("train")
t0 = time.time()
for it in range(start_iter, args.max_iters + 1):
    if it % args.eval_interval == 0 or it == args.max_iters:
        losses, sample = evaluate()
        best_val = min(best_val, losses["val"])
        print(f"--- iter {it}: train {losses['train']:.3f} | val {losses['val']:.3f}\n{sample}\n---")
        if it > start_iter:
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "config": asdict(cfg),
                        "iter": it, "best_val": best_val}, ckpt_path)
        t0 = time.time()
    if it == args.max_iters:
        break

    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    for _ in range(args.grad_accum):
        with autocast:
            _, loss = model(x, y)
        x, y = get_batch("train")  # fetch the next batch while the GPU works on backward
        (loss / args.grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if it % 20 == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / (it % args.eval_interval or 1)
        eta = dt * (args.max_iters - it) / 60
        print(f"iter {it:5d} | loss {loss.item():.3f} | lr {lr_at(it):.2e} | "
              f"{tokens_per_step / dt / 1e3:.0f}k tok/s | eta {eta:.0f} min")
