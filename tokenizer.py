"""Byte-level BPE tokenizer, trained from scratch (same idea as GPT-2's).

Text -> UTF-8 bytes (ids 0-255). Training repeatedly finds the most frequent
adjacent pair of ids and merges it into a new id (256, 257, ...), until the
vocab is full. One extra id at the end is the end-of-text token that separates
stories.

Text is first split into "words" with a regex, so merges never cross a word
boundary ("dog." and "dog," share the "dog" token).
"""
import json
import re
from collections import Counter, defaultdict

# English-only split: contractions, words with their leading space, numbers,
# punctuation runs, whitespace. GPT-2 uses \p{L} via the `regex` package;
# TinyStories is English, so stdlib `re` with A-Za-z is enough.
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def _merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(word_counts, vocab_size):
    """word_counts: Counter of pre-split words. Returns the ordered merge list."""
    words = [list(w.encode("utf-8")) for w in word_counts]
    freqs = list(word_counts.values())
    pair_counts = Counter()
    where = defaultdict(set)  # pair -> indices of words that (may) contain it
    for wi, ids in enumerate(words):
        for p in zip(ids, ids[1:]):
            pair_counts[p] += freqs[wi]
            where[p].add(wi)

    merges = []
    for new_id in range(256, vocab_size - 1):  # last id is reserved for end-of-text
        pair = max(pair_counts, key=pair_counts.get, default=None)
        if pair is None or pair_counts[pair] < 2:
            break
        merges.append(pair)
        # Only words containing the pair change; update their pair counts in place
        # instead of recounting the whole corpus every merge.
        for wi in where.pop(pair):
            ids, f = words[wi], freqs[wi]
            for p in zip(ids, ids[1:]):
                pair_counts[p] -= f
            ids = words[wi] = _merge(ids, pair, new_id)
            for p in zip(ids, ids[1:]):
                pair_counts[p] += f
                where[p].add(wi)
        del pair_counts[pair]
    return merges


class Tokenizer:
    def __init__(self, merges):
        self.merges = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.merges.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.eot = 256 + len(self.merges)
        self.vocab_size = self.eot + 1
        self._cache = {}

    def _encode_word(self, word):
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge first, exactly as in training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        out = []
        for word in PAT.findall(text):
            ids = self._cache.get(word)
            if ids is None:
                ids = self._cache[word] = self._encode_word(word)
            out.extend(ids)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids if i != self.eot).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": list(self.merges)}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])


if __name__ == "__main__":
    text = "Once upon a time, a little dog named Max didn't like the rain. Max's friend was a cat. Héllo 123!"
    tok = Tokenizer(train_bpe(Counter(PAT.findall(text * 20)), 300))
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round trip failed"
    assert len(ids) < len(text.encode("utf-8")), "merges did not compress"
    assert "".join(PAT.findall(text)) == text, "pre-split lost characters"
    print(f"ok: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens, vocab {tok.vocab_size}")
