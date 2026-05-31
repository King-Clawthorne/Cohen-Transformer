import json
import threading
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Optional, Iterable


def train_or_load_bpe(corpus_iter, vocab_size: int, save_path: str):
    """
    Train a byte-level BPE tokenizer on `corpus_iter` (yields strings) and
    cache it at save_path. Reuses the cached file on subsequent runs.

    Reserves a single special token <|endoftext|> at id 0.
    Returns a `tokenizers.Tokenizer` instance.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as BLPre
    from tokenizers.decoders import ByteLevel as BLDec

    p = Path(save_path)
    if p.exists():
        print(f"Loading cached tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    print(f"Training BPE tokenizer (vocab={vocab_size}) on corpus...")
    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = BLPre(add_prefix_space=False)
    tok.decoder = BLDec()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        initial_alphabet=BLPre.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(corpus_iter, trainer=trainer)
    p.parent.mkdir(parents=True, exist_ok=True)
    tok.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab={tok.get_vocab_size()})")
    return tok

def pretokenize_to_bin(
    docs_iter: Iterable[str],
    tokenizer,
    eot_id: int,
    out_path: str,
    dtype=np.uint16,
    batch_size: int = 1024,
    max_tokens: Optional[int] = None,
    log_every_docs: int = 100_000,
):
    """Stream `docs_iter`, tokenize in batches, and append token ids to a flat
    binary file at `out_path`. Writes a sidecar `<out_path>.json` with dtype +
    token count so readers can mmap without guessing.

    uint16 fits any vocab up to 65,535 (the project's 32k BPE fits with room).
    Bump to uint32 if you ever train a >64k vocab.
    """
    np_dtype = np.dtype(dtype)
    if np_dtype == np.uint16 and tokenizer.get_vocab_size() > 65535:
        raise ValueError("Vocab exceeds uint16 range; pass dtype=np.uint32.")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    docs_seen = 0
    batch: list[str] = []

    def flush(f):
        nonlocal total
        if not batch:
            return
        ids: list[int] = []
        for e in tokenizer.encode_batch(batch):
            ids.extend(e.ids)
            ids.append(eot_id)
        arr = np.asarray(ids, dtype=np_dtype)
        if max_tokens is not None and total + len(arr) > max_tokens:
            arr = arr[: max_tokens - total]
        f.write(arr.tobytes())
        total += len(arr)
        batch.clear()

    with open(p, "wb") as f:
        for text in docs_iter:
            text = text.strip()
            if not text:
                continue
            batch.append(text)
            docs_seen += 1
            if len(batch) >= batch_size:
                flush(f)
                if docs_seen % log_every_docs < batch_size:
                    print(f"  pretokenized {docs_seen:>9,} docs | {total/1e6:>8.1f}M tokens")
            if max_tokens is not None and total >= max_tokens:
                break
        flush(f)

    meta = {"dtype": np_dtype.name, "num_tokens": total, "vocab_size": tokenizer.get_vocab_size()}
    Path(str(p) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {total/1e6:.1f}M tokens ({docs_seen:,} docs) to {out_path}")
    return total


def load_pretokenized_meta(bin_path: str) -> dict:
    """Read the sidecar metadata written by `pretokenize_to_bin`."""
    meta_path = Path(str(bin_path) + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"No sidecar metadata at {meta_path}")
    return json.loads(meta_path.read_text())


def init_weights(module: nn.Module, std: float = 0.02):
    """GPT-2 style weight initialization."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    checkpoint_path: str,
    **extra,
):
    """Save model, optimizer state, and training step to checkpoint.

    `optimizer` may be a single Optimizer or a list/tuple of them (e.g. a
    Muon + AdamW hybrid). The list form is stored under
    `optimizer_state_dicts` so it round-trips cleanly through `load_checkpoint`.
    """
    if isinstance(optimizer, (list, tuple)):
        opt_payload = {"optimizer_state_dicts": [o.state_dict() for o in optimizer]}
    else:
        opt_payload = {"optimizer_state_dict": optimizer.state_dict()}
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **opt_payload,
        "step": step,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at step {step}")


class AsyncCheckpointer:
    """Non-blocking checkpoint writer.

    Each `save()` call:
      1. Snapshots model + optimizer state dicts to pinned CPU memory on the
         main thread (this is the only part that has to be synchronous — it's
         the GPU->host copy and it must capture a consistent state before
         training mutates weights for step N+1).
      2. Hands the CPU snapshot to a background thread that runs `torch.save`.

    The next `save()` call (or `close()` at shutdown) waits for the previous
    write to finish before kicking off a new one — so at most one outstanding
    write exists at a time, and a slow disk applies backpressure to training
    rather than silently piling up snapshots in RAM.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    @staticmethod
    def _snapshot_state_dict(sd: dict) -> dict:
        """Move every tensor in a state dict to CPU. Non-tensor values pass
        through unchanged (optimizer step counters, betas, etc.)."""
        out = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().to("cpu", copy=True, non_blocking=False)
            elif isinstance(v, dict):
                out[k] = AsyncCheckpointer._snapshot_state_dict(v)
            elif isinstance(v, (list, tuple)):
                snap = [
                    AsyncCheckpointer._snapshot_state_dict(x) if isinstance(x, dict)
                    else (x.detach().to("cpu", copy=True) if isinstance(x, torch.Tensor) else x)
                    for x in v
                ]
                out[k] = type(v)(snap)
            else:
                out[k] = v
        return out

    def wait(self):
        """Block until any pending write finishes; re-raise its error if any."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err

    def save(
        self,
        model: nn.Module,
        optimizer,
        step: int,
        checkpoint_path: str,
        **extra,
    ):
        # Backpressure: only one outstanding write at a time.
        self.wait()

        # Synchronous CPU snapshot — small fraction of total save time.
        model_sd = self._snapshot_state_dict(model.state_dict())
        if isinstance(optimizer, (list, tuple)):
            opt_payload = {
                "optimizer_state_dicts": [
                    self._snapshot_state_dict(o.state_dict()) for o in optimizer
                ]
            }
        else:
            opt_payload = {
                "optimizer_state_dict": self._snapshot_state_dict(optimizer.state_dict())
            }

        payload = {"model_state_dict": model_sd, **opt_payload, "step": step}
        if extra:
            payload.update(extra)

        def _writer():
            try:
                tmp = checkpoint_path + ".tmp"
                torch.save(payload, tmp)
                # Atomic rename so a crash mid-write never leaves a corrupt
                # checkpoint in place of the previous good one.
                Path(tmp).replace(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path} at step {step}")
            except BaseException as e:
                self._error = e

        self._thread = threading.Thread(target=_writer, daemon=True, name="ckpt-writer")
        self._thread.start()

    def close(self):
        self.wait()

def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str,
    device: str,
    return_checkpoint: bool = False,
):
    """Load model, optimizer state, and training step from checkpoint.

    `optimizer` may be a single Optimizer or a list/tuple. If the checkpoint's
    optimizer format does not match (e.g. resuming an AdamW-only checkpoint
    after swapping in a Muon hybrid), the optimizer state is skipped with a
    warning and only the model weights and step counter are restored.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        if isinstance(optimizer, (list, tuple)):
            states = checkpoint.get("optimizer_state_dicts")
            if states is not None and len(states) == len(optimizer):
                for o, s in zip(optimizer, states):
                    o.load_state_dict(s)
            else:
                print(
                    "Warning: checkpoint optimizer format does not match current "
                    "optimizer setup; skipping optimizer state load."
                )
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}")
    if return_checkpoint:
        return step, checkpoint
    return step

def load_hf_dataset(dataset_name: str = "wikitext", config_name: str = None, split: str = "train"):
    """Load a dataset from Hugging Face and combine text."""
    print(f"Loading {dataset_name} ({split}) from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name, split=split)
    
    if "text" not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'text' column")

    text = "\n".join(dataset["text"])

    print(f"Loaded {len(text)} characters")
    return text

def pre_chunk_data(data: torch.Tensor, block_size: int):
    """Slice data into non-overlapping chunks of block_size."""
    n_chunks = (len(data) - 1) // block_size
    trimmed_len = n_chunks * block_size + 1
    trimmed = data[:trimmed_len]
    x = trimmed[:-1].view(n_chunks, block_size)
    y = trimmed[1:].view(n_chunks, block_size)
    return x, y
