# Project State

> Updated at the end of every session, by whichever agent was driving.
> Keep it under a page. This is a baton, not a diary.

**Last updated:** 2026-10-18 by claude-code

## Where things stand

Done and working. The first full run finished on 2026-10-18: 8000 steps, train 1.232 / **val 1.242**. `out/ckpt.pt` is the trained model, and `uv run sample.py` writes coherent short stories that keep their characters consistent. The run took three sessions (steps 0-2000, 2000-6940, 6500-8000) because the app restarted and the laptop slept. Each resume picked up from the last checkpoint. The README has the results, the loss curve, samples, and what went wrong.

Code and docs are on branch `feat/initial-gpt`. There is no remote yet.

## In progress

- Nothing.

## The exact next step

Pick one of these:
1. **Publish:** run `/security-review` on the branch, create the GitHub repo, then push. Optionally attach `out/ckpt.pt` (330 MB; weights only would be about 110 MB) to a release rather than committing it.
2. **Experiment:** each idea is one flag or a few lines. Try a longer run (`--max_iters 16000`, about a full epoch, roughly 3.5 h), a bigger model (n_layer 12, check VRAM with `torch.cuda.max_memory_allocated()`), or an evaluation script that scores the model on held-out stories.

## Open questions

- Whether to push to GitHub. Run `/security-review` first (it's required before any push).

## Known traps

- Ollama keeps models in VRAM for 2 h (`OLLAMA_KEEP_ALIVE=2h`). Run `ollama ps` and then `ollama stop <model>` before training, or the 6 GB fills up.
- Windows console is cp1252. `train.py` and `sample.py` reconfigure stdout to UTF-8, because an untrained model prints U+FFFD, which cp1252 can't encode.
- `data/valid.txt` from Hugging Face starts mid-story ("u don't have to be scared..."). That comes from the upstream file and isn't a bug here.
- A run started in the Claude app's terminal panel dies when the app restarts. That happened at step 2000. Long runs go in a detached process instead, which is how the current run was started, logging to `out/train.log` (watch it with `Get-Content out\train.log -Wait`):
  `Start-Process uv -ArgumentList 'run','train.py','--resume' -RedirectStandardOutput out\train.log -RedirectStandardError out\train.err.log -WindowStyle Hidden` (set `$env:PYTHONUNBUFFERED='1'` first, or the log only fills when the buffer flushes).
- Resume used to load the checkpoint onto the GPU, which pushed VRAM to its limit and dropped speed from 34k to 7k tok/s. It now loads to CPU (`train.py`, `sample.py`). If speed collapses again, look at VRAM first.
