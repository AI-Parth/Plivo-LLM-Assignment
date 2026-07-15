# NOTES

Best configuration: byte-level BPE tokenizer (vocab 4096, trained only on
train_corpus.txt, ~3.9 bytes/token) feeding a RoPE + RMSNorm + SwiGLU
transformer (8 layers, n_embd=120, 5 heads, tied embeddings, no biases,
1,991,160 params) trained with AdamW (wd=0.1 on 2D weights only), linear
warmup (5%) into cosine decay to 10% of peak lr=8e-4, grad clipping at 1.0,
batch=32, block_size=128, for the full 2000 optimizer steps.

It works better than the baseline for three independent reasons. First, the
BPE tokenizer roughly quadruples how much raw text fits in the same
128-token window, which matters a lot here because ~14% of the corpus is
Devanagari and byte tokenization was burning 3 tokens per Hindi character.
Second, removing the learned position table and untying nothing (weight
tying) freed enough of the 2M parameter budget to go from the baseline's
4 layers to 8 while staying under the cap, and RoPE plus the GPT-2-style
scaled residual init kept that extra depth trainable in only 2000 steps.
Third, and the biggest single win in the RUNLOG sweeps: the step count is
capped but batch size is not, so a bigger batch buys a better gradient
estimate "for free" — going from batch 8 to 32 alone took dev bpb from
2.36 to 2.22 at a fixed 300 steps, which is why the final run uses batch=32
instead of the baseline's 8. Longer sequences (block_size 192) were tried
too but didn't beat spending the same extra compute on batch size.
