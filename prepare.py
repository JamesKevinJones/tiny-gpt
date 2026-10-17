"""Download TinyStories, train the tokenizer, and write token ids to data/*.bin.

Run once:  uv run prepare.py
Produces:  data/tokenizer.json, data/train.bin, data/val.bin (uint16 token ids)
"""
import multiprocessing as mp
import os
import time
import urllib.request
from collections import Counter

import numpy as np

from tokenizer import PAT, Tokenizer, train_bpe

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB_SIZE = 4096  # small vocab keeps the embedding/logits cheap on a 6 GB GPU
TOKENIZER_SAMPLE_CHARS = 100_000_000  # train BPE on the first ~100 MB of stories
EOT = "<|endoftext|>"


def download(split):
    path = os.path.join(DATA, f"{split}.txt")
    if os.path.exists(path):
        return path

    def progress(blocks, block_size, total):
        print(f"\r  {split}: {blocks * block_size / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(URL.format(split), path + ".part", progress)
    os.replace(path + ".part", path)  # only a finished download gets the real name
    print()
    return path


def chunks(path, size=1 << 24):
    """Yield ~16M-char pieces of the file, each cut on a story boundary."""
    with open(path, encoding="utf-8", errors="replace") as f:
        rest = ""
        while block := f.read(size):
            block = rest + block
            cut = block.rfind(EOT)
            if cut == -1:
                rest = block
                continue
            yield block[:cut]
            rest = block[cut + len(EOT):]
        if rest.strip():
            yield rest


_tok = None


def _init_worker(tokenizer_path):
    global _tok
    _tok = Tokenizer.load(tokenizer_path)


def _encode_chunk(text):
    ids = []
    for story in text.split(EOT):
        story = story.strip()
        if story:
            ids.extend(_tok.encode(story))
            ids.append(_tok.eot)
    return np.array(ids, dtype=np.uint16)


def main():
    os.makedirs(DATA, exist_ok=True)
    train_txt, val_txt = download("train"), download("valid")

    tok_path = os.path.join(DATA, "tokenizer.json")
    if not os.path.exists(tok_path):
        t0 = time.time()
        counts, seen = Counter(), 0
        for text in chunks(train_txt):
            for story in text.split(EOT):
                counts.update(PAT.findall(story.strip()))
            seen += len(text)
            if seen >= TOKENIZER_SAMPLE_CHARS:
                break
        print(f"tokenizer: {len(counts):,} unique words in {seen / 1e6:.0f}M chars, training BPE...")
        Tokenizer(train_bpe(counts, VOCAB_SIZE)).save(tok_path)
        print(f"tokenizer: done in {time.time() - t0:.0f}s")

    for txt, name in [(val_txt, "val"), (train_txt, "train")]:
        out = os.path.join(DATA, f"{name}.bin")
        if os.path.exists(out):
            continue
        t0, n = time.time(), 0
        workers = max(1, os.cpu_count() - 2)
        with mp.Pool(workers, _init_worker, (tok_path,)) as pool, open(out + ".part", "wb") as f:
            for arr in pool.imap(_encode_chunk, chunks(txt)):  # imap keeps story order
                arr.tofile(f)
                n += len(arr)
                print(f"\r  {name}: {n / 1e6:,.1f}M tokens", end="", flush=True)
        os.replace(out + ".part", out)
        print(f"\n  {name}: {n:,} tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
