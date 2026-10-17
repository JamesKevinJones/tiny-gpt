# tiny-gpt

The canonical context file. Claude Code, Antigravity (`agy`), and Codex all read
this, directly or by import. Durable project knowledge goes here. Session status
goes in `docs/STATE.md`.

## What this is

A GPT-2 style language model written and trained from scratch on one laptop
(i5-13420H, 16 GB RAM, RTX 4050 Laptop 6 GB VRAM). It includes its own byte-level BPE
tokenizer, a decoder-only transformer, and a training loop. It trains on
TinyStories and learns to write short, simple children's stories. It's a learning
project: the code is meant to be read, so clarity beats cleverness.

## Stack

- Python 3.12 (pinned in `pyproject.toml`; the system Python 3.14 has no trusted CUDA torch wheels)
- PyTorch with CUDA 13.0 wheels from `download.pytorch.org/whl/cu130`, managed by `uv`
- numpy for the uint16 token files. No other dependencies.
- Data: TinyStoriesV2-GPT4 from Hugging Face (`roneneldan/TinyStories`)

## Layout

```
tokenizer.py   BPE train/encode/decode; `python tokenizer.py` is its self-check
prepare.py     download data, train tokenizer, write data/{train,val}.bin
model.py       the transformer
train.py       training loop, writes out/ckpt.pt at every eval
sample.py      generate text from out/ckpt.pt
data/, out/    generated, git-ignored (about 3.5 GB)
```

## Rules

1. Match the surrounding code: naming, structure, comment density. Comments explain *why*, for a learner.
2. Don't add dependencies without asking. No HF `datasets`/`transformers`/`tiktoken`; the point is from scratch.
3. Keep it fitting in 6 GB VRAM. Check `torch.cuda.max_memory_allocated()` after changing model or batch size.
4. Run the checks in `docs/VERIFY.md` before reporting work as done.
5. Don't restructure things you weren't asked to touch.

## System Operating Modes

Each mode is a persona defined in `.claude/modes/`. It sets what to focus on,
how to judge the work, and the output format.

| Mode | File | Switch (Claude Code) | Badge |
| --- | --- | --- | --- |
| Business Analyst | `ba.md` | `/mode ba` or `/ba` | `[Mode: Business Analyst]` |
| System Architect | `architect.md` | `/mode architect` or `/architect` | `[Mode: System Architect]` |
| Engineer (**default**) | `engineer.md` | `/mode engineer` or `/code` | `[Mode: Engineer]` |
| Auditor | `auditor.md` | `/mode auditor` or `/audit` | `[Mode: Auditor]` |

- `/mode reset` returns to Engineer.
- **Start every response with the current mode's badge on its own line.** If no mode has been chosen this session, use `[Mode: Engineer]`.
- A mode lasts until it is switched or reset. The Rules above apply in every mode.
- Codex and `agy` don't have these slash commands. Say "switch to ba mode" and they read `.claude/modes/ba.md` directly.

## Read these too

- `docs/STATE.md`: where we stopped, what's next
- `docs/DECISIONS.md`: why things are the way they are
- `docs/VERIFY.md`: how to prove a change works

## Don't touch

- `data/`, `out/`: generated. Delete `data/*.bin` (and `tokenizer.json` if the vocab changes) to rebuild.
