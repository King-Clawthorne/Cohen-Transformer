import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset


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
