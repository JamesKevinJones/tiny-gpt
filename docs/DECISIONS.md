# Decisions

Append-only. Newest at the top. Never edit an old entry. If it stops being
true, add a new one that supersedes it and say so.

The point is to stop a fresh agent from "fixing" something you chose
deliberately. If a choice would look wrong without context, it belongs here.

---

## 2026-10-17: Model is 8 layers x 512 wide, 512-token context (~27M params)

**Context.** The RTX 4050 Laptop has 6 GB VRAM and in practice reaches about 7 TFLOPS in bf16 on this model. The training budget is about 2 hours. Measured: 38k tok/s, 3.4 GiB peak VRAM at batch 32 (on AC, Ultimate Performance).

**Decision.** n_layer=8, n_head=8, n_embd=512, block_size=512, bf16 autocast, about 32k tokens per optimizer step, 8k steps (about 260M tokens, about half a pass over TinyStories). That gives roughly 10 tokens of data per parameter, a bit under the ~20 that is compute-optimal.

**Why not bigger.** A 100M+ model would fit in memory, but it would need about 4x the tokens to train properly. That's 6+ hours on this GPU, and TinyStories doesn't have that many tokens, so it would repeat data.

**Consequences.** Stories are coherent for a paragraph or so. Don't expect general knowledge or instruction following.

---

## 2026-10-17: Own BPE tokenizer with a 4096 vocab, not tiktoken GPT-2 (50257)

**Context.** The project is "from scratch". Also, with GPT-2's vocab, the embedding is 26M of a 27M-param model's weights, and the B x T x 50257 logits tensor alone takes about 1.6 GB in bf16.

**Decision.** A byte-level BPE in `tokenizer.py`, trained on the first 100 MB of TinyStories, with 4096 ids (the last one is end-of-text). This matches what llama2.c used for TinyStories.

**Why not tiktoken.** It's an extra dependency and a black box, and a 50k vocab wastes the parameter and VRAM budget on embeddings for words TinyStories never uses.

**Consequences.** The pre-split regex is ASCII-only (stdlib `re` has no `\p{L}`). Non-English text still round-trips through bytes, but compresses badly. Changing `VOCAB_SIZE` means deleting `data/tokenizer.json` and `data/*.bin` and retraining.

---

## 2026-10-17: No torch.compile

**Context.** On Windows, torch.compile needs Triton, which official PyTorch wheels don't ship.

**Decision.** Run in eager mode. Attention already uses the fused `scaled_dot_product_attention` kernel.

**Why not triton-windows.** It's a third-party build, for a speedup of maybe 1.3x. Revisit if training time starts to matter.

**Consequences.** None functionally. Only throughput.
