import torch
import torch.nn as nn
import torch.distributed.tensor  # ensure DTensor is loaded before liger-kernel checks for it
from typing import Optional, Tuple

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.ops.rope import LigerRopeFunction

class RMSNorm(LigerRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(hidden_size=dim, eps=eps)

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotary positional encoding to queries and keys for relative position awareness.
    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # [seq_len, dim/2] -> [1, seq_len, dim]  (LigerRopeFunction expects [1, seq, dim])
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, :])
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, offset: int = 0, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos, sin: [1, seq_len, head_dim] for LigerRopeFunction.
        """
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)

        if position_ids is not None:
            cos = self.cos_cached[0][position_ids]  # [batch, seq_len, dim]
            sin = self.sin_cached[0][position_ids]
            return cos, sin
        else:
            return self.cos_cached[:, offset : offset + seq_len, :], self.sin_cached[:, offset : offset + seq_len, :]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies rotary position embeddings to q and k via fused Liger kernel."""
    return LigerRopeFunction.apply(q, k, cos, sin)
