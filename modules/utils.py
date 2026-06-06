import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict_saver import async_save
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
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

def save_checkpoint_async(
    model: nn.Module,
    optimizers,
    step: int,
    checkpoint_id: str,
    prev_future=None,
):
    """Save model + optimizer state + step using PyTorch's built-in async
    Distributed Checkpoint (`dcp.async_save`).

    `async_save` stages the state dict (to CPU) synchronously and writes it to
    disk on a background thread, so the training loop is not blocked by I/O —
    replacing the previous hand-rolled snapshot-and-thread logic. It returns a
    Future; pass it back in as `prev_future` on the next call so we wait for the
    previous write to finish before staging a new one.

    `optimizers` may be a single Optimizer or a list/tuple (Muon + AdamW hybrid).
    `get_state_dict` produces optimizer state that round-trips cleanly into fresh,
    un-stepped optimizers via `set_state_dict` in `load_checkpoint`.
    """
    if prev_future is not None:
        prev_future.result()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    model_sd, optim_sd = get_state_dict(model, optimizers)
    future = async_save(
        {"model": model_sd, "optims": optim_sd, "step": step},
        checkpoint_id=checkpoint_id,
    )
    print(f"Checkpoint staged for {checkpoint_id} at step {step} (writing in background)")
    return future


def load_checkpoint(
    model: nn.Module,
    optimizers,
    checkpoint_id: str,
    device: str = None,
):
    """Load model, optimizer state, and training step from a DCP checkpoint.

    DCP loads in place, so we seed the load with the current (possibly empty)
    state dicts via `get_state_dict`, run `dcp.load`, then apply the loaded
    tensors back onto the live model/optimizers with `set_state_dict` — which
    correctly materializes optimizer state into fresh optimizers. Returns the
    saved step. `device` is accepted for backward compatibility and unused (DCP
    places tensors to match the live model).
    """
    if not Path(checkpoint_id).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_id}")
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    model_sd, optim_sd = get_state_dict(model, optimizers)
    state = {"model": model_sd, "optims": optim_sd, "step": 0}
    dcp.load(state, checkpoint_id=checkpoint_id)
    set_state_dict(
        model,
        optimizers,
        model_state_dict=state["model"],
        optim_state_dict=state["optims"],
    )
    step = state["step"]
    print(f"Checkpoint loaded from {checkpoint_id}, resuming from step {step}")
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
