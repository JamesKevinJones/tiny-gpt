"""A GPT-2 style decoder-only transformer.

tokens -> token embedding + position embedding
       -> n_layer x Block( x + Attention(LayerNorm(x)),  x + MLP(LayerNorm(x)) )
       -> LayerNorm -> linear to vocab (weights shared with the token embedding)
       -> logits: a score for every possible next token, at every position
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512  # max context length in tokens
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0  # one pass over the data barely overfits, so no dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size): each head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused kernel for exactly this:
        #   att = (q @ k.transpose(-2, -1)) / sqrt(head_size)   how much each token cares about each other
        #   att = att.masked_fill(future_positions, -inf)        a token may not look ahead
        #   y   = softmax(att, dim=-1) @ v                       weighted average of the values
        # It never materializes the T x T matrix, which is what makes it fit in 6 GB.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate the heads back
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Attention lets tokens exchange information; the MLP then processes each
        # token on its own. Both are added to the residual stream x, not replacing it.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: same matrix reads and writes tokens

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream, so its
        # variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        # Next-token prediction: logits at position t are scored against token t+1.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, device_type):
        # Weight-decay the matrices only; the 1-D LayerNorm weights are left alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=device_type == "cuda")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_token=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / temperature  # only the last position predicts what's next
            if top_k:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = -float("inf")  # sample only among the k likeliest
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if stop_token is not None and nxt.item() == stop_token:
                break
            idx = torch.cat((idx, nxt), dim=1)
        return idx
