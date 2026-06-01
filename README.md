# Cohen-Transformer

## Situation

Cohen-Transformer is a compact PyTorch decoder-only transformer language model with a modern training stack: a custom BPE tokenizer, streaming pretraining on FineWeb-Edu, a Muon + AdamW hybrid optimizer, and KV-cache-accelerated generation. It is designed to fit a ~100M-parameter model on a single Blackwell-class GPU while keeping the implementation small enough to read end to end.

## Task

Provide a clear, concise project overview and usage guide that helps a developer understand the model's architecture, the training pipeline, and how to run or extend it.

## Action

- Implemented a minimal ~100M-parameter decoder-only transformer in `simple.py` with multi-head attention, SwiGLU MLP, RMSNorm, rotary position embeddings (RoPE), LayerScale residual gating, QK-gain, logit softcap, weight-tied embeddings/lm_head, and KV-cache-aware generation with top-k / top-p / min-p / repetition-penalty samplers.
- Switched the optimizer to `torch.optim.Muon` for the body's 2D weight matrices (Newton-Schulz–orthogonalized momentum with the Moonshot `match_rms_adamw` LR adjustment) paired with AdamW for embeddings, RMSNorm gains, and LayerScale vectors.
- Swapped training data from TinyStories to a streaming `HuggingFaceFW/fineweb-edu` `sample-10BT` pipeline: per-worker shard splitting on an `IterableDataset`, an on-the-fly tokenized held-out val set, and a 32k custom BPE trained on a 50k-doc sample.
- Runs attention through `F.scaled_dot_product_attention` with the cuDNN fused backend prioritized via `sdpa_kernel(..., set_priority=True)`, folding the learnable per-head QK-gain into `q` so it scales the score before softmax. Training/prefill uses the fast causal flag; KV-cache decode swaps in an explicit lower-right causal mask.
- Centralized tokenizer training and (multi-optimizer) checkpoint save/load in `modules/utils.py`; core building blocks (`RMSNorm`, `RotaryEmbedding`, `apply_rope`) in `modules/layers.py`.

## Result

- A self-contained training script that reads top-to-bottom, runs on a single Blackwell-class GPU, and reaches modern recipe parity (Muon + FineWeb-Edu + BF16/FP8 + `torch.compile` + cuDNN SDPA) without any framework on top of PyTorch.
- Tokenizer cache, multi-optimizer checkpoint format, streaming dataset, and an FP8-friendly model definition all reusable for further experiments.
- Solid foundation for trying architectural variants (GQA, deeper stacks, alternative residual schemes) against a known-good baseline.

## Quick Start

1. Install dependencies (PyTorch nightly recommended for Blackwell / cuDNN SDPA; `torchao` only needed for `--fp8`):

   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install datasets tokenizers
   pip install torchao  # optional, for --fp8
   ```

2. (Optional, one-time) Pre-tokenize FineWeb-Edu to a flat `.bin` so future runs skip in-loader tokenization:

   ```bash
   pip install -r requirments.txt
   ```

3. Train (auto-uses `.bin` files if present; otherwise streams + tokenizes on the fly):

   ```bash
   python simple.py --max-steps 9999 --batch-size 99 --block-size 2048
   ```

4. Train with FP8 body matmuls on Blackwell (~1.5–1.8× step-time speedup):

   ```bash
   python simple.py --fp8 --max-steps 9999 --batch-size 192 --block-size 2048
   ```

5. Resume from a checkpoint and skip straight to generation:

   ```bash
   python simple.py --resume --max-steps 0 --prompt "The capital of France"
   ```

## Files

- `simple.py` — model, training loop, FineWeb-Edu streaming dataset, FP8 wiring, cuDNN SDPA path, and generation
- `modules/layers.py` — `RMSNorm`, `RotaryEmbedding`, `apply_rope`
- `modules/utils.py` — BPE training/loading, checkpoint save/load (single or list of optimizers), dataset helpers

## Things I want to add in the future:
1. Flex attention with document masking
2. A better design for the residual stream
