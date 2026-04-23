import math
import torch
import torch.nn as nn


class MQA(nn.Module):
    def __init__(self, dim, num_head, max_seq_len=4096):
        super().__init__()
        assert dim % num_head == 0

        self.dim = dim
        self.num_head = num_head
        self.head_dim = dim // num_head

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.o_proj = nn.Linear(dim, dim)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),  # (max_seq_len, max_seq_len)
        )

    def forward(self, x):
        b, n, _ = x.shape  # x: (b, n, dim)
        h, d = self.num_head, self.head_dim

        q = self.q_proj(x).view(b, n, h, d).transpose(1, 2)  # (b, h, n, d)
        k = self.k_proj(x).unsqueeze(1)  # (b, 1, n, d), shared across heads
        v = self.v_proj(x).unsqueeze(1)  # (b, 1, n, d), shared across heads

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # (b, h, n, n), attention logits
        mask = self.causal_mask[:n, :n].view(1, 1, n, n)  # (1, 1, n, n), causal mask
        score = torch.softmax(attn + mask, dim=-1)  # (b, h, n, n), attention weights

        out = score @ v  # (b, h, n, d)
        out = out.transpose(1, 2).contiguous().view(b, n, self.dim)  # (b, n, dim)
        return self.o_proj(out)  # (b, n, dim)


if __name__ == "__main__":
    batch, len_seq, dim = 10, 300, 32
    model = MQA(dim, num_head=8)
    x = torch.randn(batch, len_seq, dim)
    y = model(x)
    print(y.shape)
    assert x.shape == y.shape
