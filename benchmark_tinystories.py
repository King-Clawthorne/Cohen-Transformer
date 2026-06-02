import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer


def load_checkpoint_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain a model_state_dict")
    return checkpoint, normalize_state_dict(checkpoint["model_state_dict"])


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("_orig_mod.") or key.startswith("module."):
            key = key.split(".", 1)[1]
        normalized[key] = value
    return normalized


def infer_model_config(state_dict):
    emb = state_dict["token_emb.weight"]
    vocab_size, n_embd = emb.shape
    n_layers = sum(
        1
        for key in state_dict
        if key.startswith("qkv.") and key.endswith(".weight")
    )
    qk_gain = state_dict.get("qk_gain.0")
    if qk_gain is None:
        raise KeyError("Could not infer n_heads because qk_gain.0 is missing")
    n_heads = qk_gain.numel()

    cos = state_dict.get("rope.cos_cached")
    if cos is not None:
        block_size = cos.shape[1]
    else:
        block_size = 2048

    return {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_embd": n_embd,
    }


def load_model(checkpoint_path, device, logit_softcap):
    try:
        from simple import SimpleTransformerLM
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise ModuleNotFoundError(
            f"Could not import simple.py because {missing!r} is not installed. "
            "Install the training requirements before running the benchmark."
        ) from exc

    checkpoint, state_dict = load_checkpoint_state(Path(checkpoint_path))
    config = infer_model_config(state_dict)
    model = SimpleTransformerLM(
        **config,
        logit_softcap=logit_softcap,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, checkpoint.get("step", None)


def iter_tinystories_texts(dataset_name, split, text_column, max_docs, streaming):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for i, example in enumerate(dataset):
        if max_docs is not None and i >= max_docs:
            break
        text = example.get(text_column)
        if text and text.strip():
            yield text.strip()


def iter_token_blocks(texts, tokenizer, eot_id, block_size):
    buffer = []
    for text in texts:
        buffer.extend(tokenizer.encode(text).ids)
        if eot_id is not None:
            buffer.append(eot_id)
        while len(buffer) >= block_size + 1:
            x = torch.tensor(buffer[:block_size], dtype=torch.long)
            y = torch.tensor(buffer[1 : block_size + 1], dtype=torch.long)
            yield x, y
            buffer = buffer[block_size:]


def iter_batches(blocks, batch_size, max_batches):
    xs, ys = [], []
    yielded = 0
    for x, y in blocks:
        xs.append(x)
        ys.append(y)
        if len(xs) == batch_size:
            yield torch.stack(xs), torch.stack(ys)
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                return
            xs, ys = [], []


def autocast_context(device, dtype_name):
    if not str(device).startswith("cuda") or dtype_name == "float32":
        return nullcontext()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def benchmark(model, batches, device, dtype_name, warmup_batches):
    total_loss = 0.0
    total_tokens = 0
    measured_batches = 0
    measured_time = 0.0

    with torch.inference_mode():
        for batch_idx, (xb, yb) in enumerate(batches):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            with autocast_context(device, dtype_name):
                logits, _ = model(xb, use_cache=False)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if batch_idx < warmup_batches:
                continue

            tokens = yb.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            measured_batches += 1
            measured_time += elapsed

    if measured_batches == 0 or total_tokens == 0:
        raise RuntimeError("No benchmark batches were produced. Try lowering --block-size or --batch-size.")

    loss = total_loss / total_tokens
    return {
        "batches": measured_batches,
        "tokens": total_tokens,
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "seconds": measured_time,
        "tokens_per_second": total_tokens / measured_time,
        "milliseconds_per_batch": 1000.0 * measured_time / measured_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a simple.py checkpoint on the TinyStories validation set."
    )
    parser.add_argument("--checkpoint", default="simple_checkpoint.pt")
    parser.add_argument("--tokenizer-path", default="fineweb_edu_bpe.json")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="default",
    )
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    if str(args.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Use the tokenizer saved during training."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    model, config, step = load_model(args.checkpoint, args.device, args.logit_softcap)
    eval_block_size = args.block_size or config["block_size"]
    if eval_block_size > config["block_size"]:
        raise ValueError(
            f"--block-size={eval_block_size} exceeds checkpoint block size {config['block_size']}"
        )

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=True)

    texts = iter_tinystories_texts(
        args.dataset,
        args.split,
        args.text_column,
        args.max_docs,
        streaming=not args.no_streaming,
    )
    blocks = iter_token_blocks(texts, tokenizer, eot_id, eval_block_size)
    batches = iter_batches(
        blocks,
        batch_size=args.batch_size,
        max_batches=args.max_batches + args.warmup_batches,
    )

    result = benchmark(model, batches, args.device, args.dtype, args.warmup_batches)

    step_text = "unknown" if step is None else str(step)
    print(f"checkpoint: {args.checkpoint} (step {step_text})")
    print(f"dataset:    {args.dataset} [{args.split}]")
    print(
        "model:      "
        f"{config['n_layers']} layers, {config['n_heads']} heads, "
        f"{config['n_embd']} dim, block {config['block_size']}, vocab {config['vocab_size']}"
    )
    print(f"eval:       batch {args.batch_size}, block {eval_block_size}, dtype {args.dtype}")
    print(f"batches:    {result['batches']}")
    print(f"tokens:     {result['tokens']:,}")
    print(f"loss:       {result['loss']:.4f}")
    print(f"perplexity: {result['perplexity']:.2f}")
    print(f"throughput: {result['tokens_per_second']:,.0f} tokens/s")
    print(f"latency:    {result['milliseconds_per_batch']:.2f} ms/batch")


if __name__ == "__main__":
    main()
