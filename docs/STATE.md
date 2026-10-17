# Project State

> Updated at the end of every session, by whichever agent was driving.
> Keep it under a page. This is a baton, not a diary.

**Last updated:** 2026-10-17 by claude-code

## Where things stand

Everything is written and smoke-tested: the tokenizer, data prep, the model, training and sampling. `data/` is built (552.6M train tokens, 5.6M val tokens, vocab 4096). A 60-step smoke run took loss from 8.43 to 4.56. The first full run (`uv run train.py`, 8000 steps, about 1 h 50 min at about 40k tok/s) was started on 2026-10-17 in a terminal tab. It writes `out/ckpt.pt` every 500 steps.

Nothing is committed yet. All work is on branch `feat/initial-gpt`, which has no commits.

## In progress

- [ ] First full training run. Check that it finished: `uv run sample.py --num 3` should print readable stories and show iter 8000.

## The exact next step

Once training finishes, record the final train/val loss and one sample in the README (there isn't one yet; write it). If the run was interrupted, use `uv run train.py --resume`.

## Open questions

- Whether to push to GitHub. Run `/security-review` first (it's required before any push).

## Known traps

- Ollama keeps models in VRAM for 2 h (`OLLAMA_KEEP_ALIVE=2h`). Run `ollama ps` and then `ollama stop <model>` before training, or the 6 GB fills up.
- Windows console is cp1252. `train.py` and `sample.py` reconfigure stdout to UTF-8, because an untrained model prints `�`, which cp1252 can't encode.
- `data/valid.txt` from Hugging Face starts mid-story ("u don't have to be scared..."). That comes from the upstream file and isn't a bug here.
- `run_in_terminal` can't pipe (`| Tee-Object`), so the training log lives only in the terminal scrollback.
