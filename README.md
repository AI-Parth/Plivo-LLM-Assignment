# 2,000-Step LLM Speedrun

A from-scratch GPT trained under a hard budget: **2,000 optimizer steps**,
**2,000,000 parameters**, CPU only, trained on a single ~7MB mixed
English + Hindi corpus (`data/train_corpus.txt`). The task and rules are in
[LLM_assignment.pdf](LLM_assignment.pdf); everything that answers it lives in
[starter/](starter/).

## TL;DR of what changed vs. the given baseline

The starter code (byte tokenizer, 4-layer GPT with learned positions, plain
Adam, constant LR) scores **2.3718 bits-per-byte** on the held-out dev file
after the full 2,000 steps. Three changes, in order of how much they
mattered:

1. **A trained byte-level BPE tokenizer** (`starter/tokenizer.py`, vocab
   4096, trained only on `train_corpus.txt`) instead of raw bytes. ~14% of
   the corpus is Devanagari, and byte tokenization burns 3 tokens per Hindi
   character. The BPE tokenizer gets ~3.9 bytes/token, so the same
   `block_size` now covers ~4x more real text. Still fully lossless
   (byte-level fallback, verified round-trip on dev text and on unseen
   scripts/emoji).
2. **A different architecture** spending the same parameter budget better:
   RoPE instead of learned positions, RMSNorm, SwiGLU MLP, tied
   embeddings, no linear biases, GPT-2-style scaled residual init. Removing
   the position table and tying the output head freed enough parameters to
   go from 4 layers to 8 while staying just under the 2,000,000 cap
   (1,991,160 used).
3. **Bigger batch size.** The hard cap is on optimizer *steps*, not on
   tokens-per-step or wall-clock time, and the assignment explicitly allows
   changing batch size. A bigger batch is a better gradient estimate at the
   same step count — in a controlled 300-step sweep, going from batch 8 to
   32 took dev bpb from 2.36 to 2.22, the single largest lever found (see
   `starter/RUNLOG.md`, Run 4).

Every experiment, hypothesis, and result is logged in
[starter/RUNLOG.md](starter/RUNLOG.md); the final configuration and reasoning
are summarized in [starter/NOTES.md](starter/NOTES.md) and
[starter/SUMMARY.html](starter/SUMMARY.html).

## Repo layout

```
LLM_assignment.pdf     the assignment brief
data/
  train_corpus.txt      the only training data allowed (mixed en/hi, ~7MB)
  dev_eval.txt           held-out text for self-scoring
starter/
  tokenizer.py           BPE tokenizer: trainer + load()/encode()/decode()
  bpe_vocab.json          the trained tokenizer (4096 merges), loaded by tokenizer.py
  model.py                GPT: RoPE + RMSNorm + SwiGLU + tied embeddings
  train.py                trainer: AdamW, warmup+cosine LR, grad clipping
  evaluate.py             official scorer (bits-per-byte) - interface unchanged
  ckpt.pt                 final checkpoint (2,000 steps, submitted result)
  RUNLOG.md               one entry per run: hypothesis / change / before-after / conclusion
  NOTES.md                final config + why, in <=10 sentences
  SUMMARY.html            generated summary of the whole run (see below)
```

## Reproducing

```bash
cd starter
# 1. train the tokenizer (only needs to be done once; writes bpe_vocab.json)
python tokenizer.py --data ../data/train_corpus.txt --vocab_size 4096

# 2. train the model (hard caps enforced by asserts in train.py)
python train.py --data ../data/train_corpus.txt --steps 2000 \
    --batch 32 --block_size 128 --lr 8e-4 --out ckpt.pt

# 3. score it - this exact command/interface is what grading runs
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

## Hard caps respected

- ≤2,000 optimizer steps for the run that produced `ckpt.pt` (checked by an
  `assert` in `train.py`, and the step count is recorded inside the
  checkpoint itself).
- ≤2,000,000 parameters (checked by an `assert` in `train.py`; final model
  is 1,991,160).
- Trained only on `data/train_corpus.txt` — the tokenizer too.
- Pure PyTorch + stdlib (the tokenizer trainer uses `heapq`/`unicodedata`
  from the standard library, no `regex`/`tokenizers` dependency).
- CPU only.

## Tools disclosure

AI coding agents(copilot) are used, research by an AI and human is done on kaggle
and other platforms.
