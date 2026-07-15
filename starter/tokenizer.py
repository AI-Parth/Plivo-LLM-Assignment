"""Byte-level BPE tokenizer, trained only on train_corpus.txt.

Base vocab is the 256 raw bytes, so any UTF-8 string round-trips exactly
even if it contains scripts/characters never seen in training - unmerged
bytes just fall back to single-byte tokens. That keeps encode/decode
lossless everywhere, which is what evaluate.py's round-trip check needs.

Pre-tokenization groups text into word-ish chunks (letters+combining marks
stay together, so a Devanagari syllable isn't shredded into one token per
byte) before BPE merges are learned/applied within each chunk. This is the
same idea as GPT-2's regex splitter, just written with unicodedata instead
of the external `regex` package, to stay inside "stdlib only".

Usage:
    python tokenizer.py --data ../data/train_corpus.txt --vocab_size 4096
    (writes bpe_vocab.json next to this file)

load() reads that file with no arguments, as train.py / evaluate.py expect.
"""
import argparse
import heapq
import json
import os
import unicodedata
from collections import Counter, defaultdict

_MERGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_vocab.json")


def _char_class(ch):
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)[0]
    if cat in ("L", "M", "N"):
        return "word"
    return "other"


def pretokenize(text):
    """Split text into chunks that concatenate back to the original text
    exactly. Letters + combining marks + digits form one run each; a single
    leading space is glued onto the run that follows it (GPT-2 style),
    which lets "the"/" the" collapse into one learned unit instead of two.
    """
    if not text:
        return []
    runs = []
    start = 0
    cls = _char_class(text[0])
    for i in range(1, len(text)):
        c = _char_class(text[i])
        if c != cls:
            runs.append((cls, text[start:i]))
            start, cls = i, c
    runs.append((cls, text[start:]))

    chunks = []
    i = 0
    while i < len(runs):
        rcls, chunk = runs[i]
        if rcls == "space" and i + 1 < len(runs):
            if len(chunk) > 1:
                chunks.append(chunk[:-1])
            chunks.append(chunk[-1] + runs[i + 1][1])
            i += 2
            continue
        chunks.append(chunk)
        i += 1
    return chunks


def _merge_ids(ids, pair, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, vocab_size, min_pair_count=2):
    """Standard word-frequency BPE trainer with incremental pair counts,
    tracked via a lazily-invalidated max-heap so we don't rescan every
    pair from scratch on every merge (that's what makes 4k merges over a
    7MB corpus finish in a couple of minutes instead of hours)."""
    assert vocab_size > 256
    word_freq = Counter(pretokenize(text))

    word_ids = []      # word_ids[w] = list of current token ids for word w
    word_freqs = []     # parallel frequency
    for chunk, freq in word_freq.items():
        word_ids.append(list(chunk.encode("utf-8")))
        word_freqs.append(freq)

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for w, ids in enumerate(word_ids):
        f = word_freqs[w]
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(w)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []   # ordered list of [a, b, new_id]
    next_id = 256

    while next_id < vocab_size and heap:
        neg_c, pair = heapq.heappop(heap)
        cur = pair_counts.get(pair, 0)
        if cur == 0 or -neg_c != cur:
            continue  # stale heap entry, a fresher one exists or was consumed
        if cur < min_pair_count:
            break

        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((pair[0], pair[1], new_id))

        affected = list(pair_to_words.get(pair, ()))
        for w in affected:
            ids = word_ids[w]
            f = word_freqs[w]
            if pair[0] not in ids or pair[1] not in ids:
                continue
            old_pairs = Counter(zip(ids, ids[1:]))
            new_ids = _merge_ids(ids, pair, new_id)
            if new_ids == ids:
                continue
            new_pairs = Counter(zip(new_ids, new_ids[1:]))
            word_ids[w] = new_ids

            for p, c in old_pairs.items():
                pair_counts[p] -= c * f
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_to_words[p].discard(w)
            for p, c in new_pairs.items():
                pair_counts[p] += c * f
                pair_to_words[p].add(w)
                heapq.heappush(heap, (-pair_counts[p], p))
            for p in old_pairs:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
        pair_counts.pop(pair, None)
        pair_to_words.pop(pair, None)

    return merges, vocab


class BPETokenizer:
    def __init__(self, merges=None, vocab=None):
        merges = merges or []
        self.ranks = {(a, b): i for i, (a, b, _) in enumerate(merges)}
        self.merge_to_id = {(a, b): nid for a, b, nid in merges}
        self.vocab = {int(k): bytes(v) for k, v in (vocab or {}).items()}
        for i in range(256):
            self.vocab.setdefault(i, bytes([i]))
        self.vocab_size = 256 + len(merges)
        self._cache = {}

    def _bpe_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pairs = list(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            new_id = self.merge_to_id[best]
            ids = _merge_ids(ids, best, new_id)
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            out.extend(self._bpe_chunk(chunk))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        merges = [[a, b, nid] for (a, b), nid in self.merge_to_id.items()]
        merges.sort(key=lambda m: self.ranks[(m[0], m[1])])
        data = {
            "type": "bpe",
            "merges": merges,
            "vocab": {str(i): list(b) for i, b in self.vocab.items() if i >= 256},
        }
        with open(path, "w") as f:
            json.dump(data, f)


class ByteTokenizer:
    """Fallback: raw UTF-8 bytes, vocab 256. Used if no trained BPE file
    is present, and kept around as the simplest possible correct baseline."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer evaluate.py / train.py use. Resolves the saved
    BPE file relative to this file, so cwd doesn't matter at grading time."""
    path = path or _MERGE_FILE
    if not os.path.exists(path):
        return ByteTokenizer()
    with open(path) as f:
        data = json.load(f)
    merges = [(a, b, nid) for a, b, nid in data["merges"]]
    vocab = {int(k): v for k, v in data["vocab"].items()}
    return BPETokenizer(merges=merges, vocab=vocab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out", default=_MERGE_FILE)
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8").read()
    print(f"training BPE on {len(text.encode('utf-8')):,} bytes, target vocab {args.vocab_size}")
    merges, vocab = train_bpe(text, args.vocab_size)
    tok = BPETokenizer(merges=merges, vocab=vocab)
    tok.save(args.out)
    print(f"saved {len(merges)} merges -> {args.out}")

    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round trip failed!"
    print(f"corpus: {len(text):,} chars -> {len(ids):,} tokens "
          f"({len(text.encode('utf-8')) / len(ids):.2f} bytes/token)")
