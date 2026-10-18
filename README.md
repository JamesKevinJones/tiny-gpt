# tiny-gpt

A GPT-2 style language model built from scratch and trained on one laptop. It
writes short children's stories:

> The little dragon was scared because the cat might be a threat. He tried to
> hide behind a tree, but the cat could not find him. Suddenly, the cat jumped
> out and said, "You can't hide from me! I am bigger and stronger than you!"
> The dragon was surprised and a little scared. He said, "I'm sorry, little cat.
> I didn't mean to scare you. I will be gentle from now on." The little dragon
> and the cat became friends, and they played together every day.

Everything that turns text into that story is in this repo: the tokenizer,
the transformer, and the training loop. PyTorch supplies tensors, autograd and
the attention kernel. There is no Hugging Face, no tiktoken, and no pretrained
weights.

## Results

| | |
|---|---|
| Model | 8 layers, 8 heads, 512 wide, 512-token context, 27.3M parameters |
| Tokenizer | byte-level BPE with a 4096-token vocab, trained on 100 MB of the dataset |
| Data | [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories): 552.6M training tokens |
| Training | 8000 steps × 32,768 tokens = 262M tokens (about half an epoch), bf16, AdamW |
| Hardware | RTX 4050 Laptop (6 GB), i5-13420H, 16 GB RAM |
| Speed | 33-47k tokens/s, 3.4 GiB peak VRAM, about 2 h of GPU time across three sessions |
| Final loss | train 1.232, **val 1.242** |

Validation loss over training:

| Step | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 |
|---|---|---|---|---|---|---|---|---|---|
| Val loss | 8.43 | 1.77 | 1.57 | 1.45 | 1.38 | 1.34 | 1.29 | 1.25 | 1.24 |

Step 0 is ln(4096) = 8.32 plus initialisation noise: a model that knows
nothing is guessing uniformly among 4096 tokens. At 1.24 it is choosing among
about e^1.24 ≈ 3.5 equally likely tokens. Train and val stay together the
whole way because the model sees each story at most once, so it can't overfit.

What extra training bought: at step 1000 the model already wrote grammatical
stories, but lost track of characters (a "little dragon" became a "little boy"
and then a "little cactus" in the same story). By step 8000, characters stay
consistent for the whole story. The remaining flaws are logic ones, such as
"she got a paper to prevent Spot from being in trouble".

## Run it

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU. It falls back to the
CPU, but training there takes days.

```bash
uv sync                    # Python 3.12 + PyTorch (CUDA 13.0 wheels)
uv run tokenizer.py        # tokenizer self-check
uv run prepare.py          # download 2.2 GB, train tokenizer, tokenize (~5 min)
uv run train.py            # ~1 h 45 min uninterrupted; writes out/ckpt.pt every 500 steps
uv run sample.py --prompt "The little dragon was scared because" --num 3
```

If training stops, `uv run train.py --resume` continues from the last
checkpoint.

## How it works

| File | What it does |
|---|---|
| [tokenizer.py](tokenizer.py) | Starts from the 256 byte values and repeatedly merges the most frequent adjacent pair into a new token (` t`+`he` → ` the`) until there are 4096. About 4 characters per token. |
| [prepare.py](prepare.py) | Downloads TinyStories, trains the tokenizer, and writes every story as `uint16` token ids to `data/*.bin` using all CPU cores. |
| [model.py](model.py) | Token and position embeddings, 8 blocks of causal self-attention plus an MLP on a residual stream, then a projection back to the vocab. The output layer shares weights with the input embedding. |
| [train.py](train.py) | Random 512-token windows, next-token cross-entropy, gradient accumulation, gradient clipping, warmup plus cosine learning rate, bf16 autocast, checkpoint and sample every 500 steps. |
| [sample.py](sample.py) | Loads a checkpoint and generates with temperature and top-k sampling. |

The design choices (why 27M parameters, why my own 4096-token tokenizer instead
of GPT-2's 50k, why no `torch.compile`) are in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Things that went wrong

- **Windows console encoding.** The untrained model emits invalid UTF-8,
  decoded as U+FFFD (the replacement character), which the default cp1252 console can't print. The scripts
  reconfigure stdout to UTF-8.
- **Training died with its terminal.** The first run ran inside an app's
  terminal panel and stopped at step 2000 when the app restarted. It was
  resumed as a detached process.
- **Resume ran 5x slower.** `torch.load(..., map_location="cuda")` put the
  330 MB checkpoint (weights plus optimizer state) on the GPU next to the live
  model. That pushed a 6 GB card to its limit, and Windows paged GPU memory into
  system RAM, so speed fell from 34k to 7k tokens/s. Loading to CPU first fixed
  it, because `load_state_dict` copies each tensor to the model's device anyway.
