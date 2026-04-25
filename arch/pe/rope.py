import math
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    For position pos and frequency index i:
        theta_i = 1 / 10000 ** (2i / dim)

    RoPE rotates each pair of hidden dimensions:
        [x_{2i}, x_{2i+1}] ->
        [x_{2i} * cos(pos * theta_i) - x_{2i+1} * sin(pos * theta_i),
         x_{2i} * sin(pos * theta_i) + x_{2i+1} * cos(pos * theta_i)]
    """

    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        theta = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))  # (dim / 2,)
        angle = pos * theta  # (max_seq_len, dim / 2)

        self.register_buffer("cos", torch.cos(angle).view(1, 1, max_seq_len, dim // 2))
        self.register_buffer("sin", torch.sin(angle).view(1, 1, max_seq_len, dim // 2))

    def forward(self, x):
        b, h, n, d = x.shape  # x: (b, h, n, dim)
        assert d == self.dim
        assert n <= self.max_seq_len

        x_even = x[..., 0::2]  # (b, h, n, dim / 2)
        x_odd = x[..., 1::2]  # (b, h, n, dim / 2)
        cos = self.cos[:, :, :n, :].to(dtype=x.dtype)  # (1, 1, n, dim / 2)
        sin = self.sin[:, :, :n, :].to(dtype=x.dtype)  # (1, 1, n, dim / 2)

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out  # (b, h, n, dim)

 
if __name__ == "__main__":
    batch, num_head, seq_len, head_dim = 10, 8, 300, 32
    rope = RoPE(head_dim)
    x = torch.ones(batch, num_head, seq_len, head_dim)
    y = rope(x)
