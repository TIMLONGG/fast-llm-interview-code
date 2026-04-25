import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """
    Original Transformer sinusoidal positional encoding.

    For position pos and frequency index i:
        w_i = 1 / 10000 ** (2i / dim)
        PE(pos, 2i)     = sin(pos * w_i)
        PE(pos, 2i + 1) = cos(pos * w_i)
    """

    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        # Standard vectorized form of w_i = 1 / 10000 ** (2i / dim).
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))  # (dim / 2,)

        pe = torch.zeros(max_seq_len, dim)  # (max_seq_len, dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, dim)

    def forward(self, x):
        b, n, d = x.shape  # x: (b, n, dim)
        assert d == self.dim
        assert n <= self.max_seq_len

        return x + self.pe[:, :n, :].to(dtype=x.dtype)


if __name__ == "__main__":
    batch, seq_len, dim = 10, 300, 32
    pe = SinusoidalPE(dim)
    x = torch.randn(batch, seq_len, dim)
    x = pe(x)
    
