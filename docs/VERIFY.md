# Verification

Exact commands to prove a change works. Any agent, any tool, no guessing.

Rule: **don't report work as done without running these.** "It should work" is
not a result.

## Install

```bash
uv sync
```

## Tokenizer self-check

```bash
uv run tokenizer.py
```

Prints `ok: ... bytes -> ... tokens`. Any assertion error means encode/decode or the BPE trainer broke.

## Data (once, about 15 min: 2.2 GB download plus tokenizing)

```bash
uv run prepare.py
```

Then check that `data/` has `tokenizer.json`, `val.bin` and `train.bin`. train.bin should be roughly 1 GB.

## Training smoke test (about 1 min)

Free the GPU first. Ollama keeps models in VRAM for 2 hours (`OLLAMA_KEEP_ALIVE=2h`), so run `ollama ps`, then `ollama stop <model>`.

```bash
uv run train.py --max_iters 60 --eval_interval 30 --eval_batches 5 --out out/smoke
```

Pass: loss falls from about 8.3 (ln 4096) to well under 6 by iter 60, a sample prints at each eval, and `out/smoke/ckpt.pt` exists.

## Full run

```bash
uv run train.py
uv run sample.py --num 3
```

Pass: val loss is about 1.3-1.5 by the end, and samples are readable, mostly grammatical stories.

## Known-failing

- None.
