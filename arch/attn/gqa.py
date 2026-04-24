import math
import torch
import torch.nn as nn


class GQA(nn.Module):
    def __init__(self, dim, num_head, num_kv_head, max_seq_len=4096):
        super().__init__()
        assert dim % num_head == 0
        assert num_head % num_kv_head == 0

        self.dim = dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = dim // num_head
        self.num_head_per_kv = num_head // num_kv_head

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_kv_head * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_head * self.head_dim)
        self.o_proj = nn.Linear(dim, dim)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),
        )

    def forward(self, x):
        b, n, _ = x.shape  # (b, n, dim)
        h, hk, d = self.num_head, self.num_kv_head, self.head_dim

        q = self.q_proj(x).view(b, n, h, d).transpose(1, 2)  # (b, h, n, d)
        k = self.k_proj(x).view(b, n, hk, d).transpose(1, 2)  # (b, hk, n, d)
        v = self.v_proj(x).view(b, n, hk, d).transpose(1, 2)  # (b, hk, n, d)

        k = k.repeat_interleave(self.num_head_per_kv, dim=1)  # (b, h, n, d)
        v = v.repeat_interleave(self.num_head_per_kv, dim=1)  # (b, h, n, d)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # (b, h, n, n)
        mask = self.causal_mask[:n, :n].view(1, 1, n, n)  # (1, 1, n, n)
        score = torch.softmax(attn + mask, dim=-1)  # (b, h, n, n)

        out = score @ v  # (b, h, n, d)
        out = out.transpose(1, 2).contiguous().view(b, n, self.dim)  # (b, n, dim)
        return self.o_proj(out)  # (b, n, dim)


if __name__ == "__main__":
    batch, len_seq, dim = 10, 300, 32
    model = GQA(dim, num_head=8, num_kv_head=2)
    x = torch.randn(batch, len_seq, dim)
    y = model(x)
    print(y.shape)
    assert x.shape == y.shape