"""Generate stories from a trained checkpoint.

    uv run sample.py
    uv run sample.py --prompt "The dragon was sad because" --num 3 --temperature 1.0
"""
import argparse
import os
import sys

import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
ROOT = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--num", type=int, default=1)
p.add_argument("--tokens", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)  # lower = safer, higher = wilder
p.add_argument("--top_k", type=int, default=50)
p.add_argument("--ckpt", default=os.path.join(ROOT, "out", "ckpt.pt"))
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.load(os.path.join(ROOT, "data", "tokenizer.json"))
# weights_only: a .pt is a pickle; refuse anything but tensors and plain data, so a
# downloaded checkpoint cannot run code. load_state_dict copies weights to the GPU.
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
model.load_state_dict(ckpt["model"])
print(f"checkpoint: iter {ckpt['iter']}, best val loss {ckpt['best_val']:.3f}\n")

prompt = torch.tensor([tok.encode(args.prompt)], device=device)
for _ in range(args.num):
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(prompt, args.tokens, args.temperature, args.top_k, stop_token=tok.eot)
    print(tok.decode(out[0].tolist()), "\n" + "-" * 60)
