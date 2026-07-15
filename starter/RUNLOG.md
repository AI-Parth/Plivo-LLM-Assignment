# RUNLOG

Baseline is the unmodified starter code: byte tokenizer (vocab 256), 4-layer
GPT with learned positions, Adam constant-LR, batch 8 / block 128.

## Run 0 — baseline, unmodified
- Hypothesis: n/a, just establishing a reference point.
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt_baseline.pt`
- Result: dev bpb **2.3718**, 1,339,840 params (only 67% of the param cap used),
  240s wall time.
- Conclusion: params and steps both have headroom, and every training-loop
  choice (constant LR, no warmup, no decay, no clipping) is a candidate for
  improvement. Also: byte tokenizer means a 3-byte Devanagari character eats 3
  positions of the model's 128-token window — worth checking how much of the
  corpus is Devanagari before doing anything else (14% of characters, more of
  the bytes).

## Run 1 — swap in a trained BPE tokenizer (vocab 4096)
- Hypothesis: byte-level BPE trained only on train_corpus.txt should shrink
  sequence length a lot (especially for Hindi), giving the model more real
  text per context window at the same `block_size`, for free.
- What changed: wrote a from-scratch BPE trainer (word-frequency counts +
  incremental heap-based merge selection, since `regex` isn't stdlib — used
  a unicodedata-based pretokenizer instead that keeps Devanagari
  letter+combining-mark clusters intact). vocab_size=4096, 3840 merges.
- Result: 3.9 bytes/token average on both train and dev (vs 1.0 for bytes).
  Verified lossless round-trip on dev text and on adversarial unseen input
  (Arabic script, emoji, ZWJ).
- Conclusion: kept it. This is the single highest-leverage change available —
  every subsequent token in the context window now represents ~4x more raw
  text, for the same `block_size`.

## Run 2 — new architecture: RoPE + RMSNorm + SwiGLU + tied embeddings
- Hypothesis: learned position embeddings and an untied output head are pure
  parameter overhead at this budget; spending those parameters on depth
  instead should help more than a wider/shallower net with a position table.
- What changed: model.py rewritten — RoPE (no position-embedding params),
  RMSNorm instead of LayerNorm, SwiGLU MLP (mult=3.0, replacing 4x GELU),
  weight tying (head shares tok_emb), no linear biases, GPT-2-style scaled
  init (std 0.02/sqrt(2*n_layer)) on residual-branch output projections.
  Swept configs to spend the freed budget: n_layer=8, n_embd=120, n_head=5
  → 1,991,160 params (99.6% of the 2,000,000 cap, vs baseline's 67%).
- Result: at just 300/2000 steps (batch=8, block=128, lr=8e-4) dev bpb was
  already **2.3637** — matching the baseline's FULL 2000-step score in 15%
  of the steps.
- Conclusion: kept it. Convergence per step is much faster than baseline.

## Run 3 — LR sweep (steps=300, batch=8, warmup 5% + cosine to 10%)
- Hypothesis: baseline's constant 3e-4 was never tuned; warmup+cosine should
  tolerate a higher peak LR.
- Result: 5e-4 → 2.3955, **8e-4 → 2.3637 (best)**, 1e-3 → 2.3699, 1.5e-3 → 2.4079.
- Conclusion: 8e-4 peak LR selected. Above ~1e-3 the run degrades — the
  zero/scaled residual init helps stability but doesn't make the model LR-
  insensitive.

## Run 4 — batch size sweep (steps=300, block=128, lr=8e-4)
- Hypothesis: the hard cap is on optimizer *steps*, not on tokens processed
  per step or wall-clock time. A bigger batch buys a better gradient
  estimate — effectively more "training" — without touching the step count
  or the parameter count, so it should be free performance within the rules
  (the brief explicitly lists batch size as changeable).
- Result: batch=8 → 2.3637, batch=16 → **2.274**, batch=24 → 2.219,
  batch=32 → 2.2227. Clear monotonic improvement, flattening out after ~24.
- Conclusion: this was the single biggest lever found. Picked batch=32 for
  the final run — batch=64 was tried but the grading machine here got
  noticeably noisy/throttled and a 300-step probe at batch=64 wasn't worth
  the wall-clock cost for a result that batch=24 vs 32 already showed was
  flattening out.

## Run 5 — block_size 192 instead of 128, batch adjusted to match tokens/step
- Hypothesis: since batch helped, does trading batch for more context
  length per sequence (longer `block_size`) help even more, at similar
  compute per step?
- What changed: batch=24, block_size=192 (4,608 tokens/step, comparable to
  batch=32/block=128's 4,096).
- Result: bpb **2.219**, essentially tied with the batch=32/block=128 result
  (2.2227) — not a meaningful difference, and it ran ~4x slower per step
  than the batch=32/block=128 config for that marginal gain.
- Conclusion: not worth it here. block_size=128 (~500 bytes of real context
  thanks to the BPE compression) is already enough context; extra batch is
  the cheaper way to spend compute. Kept block_size=128, batch=32 for the
  final run.

## Run 6 — final full run
- Config: BPE tokenizer (vocab 4096), RoPE+RMSNorm+SwiGLU+tied 8-layer /
  120-dim / 5-head GPT (1,991,160 params), AdamW (wd=0.1, decoupled from
  norms), warmup 5% + cosine to 10% of peak, grad clip 1.0, lr=8e-4,
  batch=32, block_size=128, steps=2000 (at the cap).
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --batch 32 --block_size 128 --lr 8e-4 --out ckpt.pt`
- Result: dev bpb **RESULT_PLACEHOLDER** (see NOTES.md), params 1,991,160, steps 2000.
- Conclusion: see NOTES.md for the final writeup.
