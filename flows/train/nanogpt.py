"""The model and its training loop; `PYTHONPATH=src python flows/train/nanogpt.py <preset>` trains one locally."""
from __future__ import annotations  # see config.py

import argparse
import math
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import PRESETS, Config, load_text, resolve


class Rotary(nn.Module):
    """Rotary position embedding (RoPE): rotates each (even, odd) pair of q/k channels by a position-dependent angle."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE needs an even head dim, got {head_dim}"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)  # (max_len, head_dim/2)
        # buffers, not parameters: no weights to learn, and kept out of the state dict
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is (B, n_head, T, head_dim)."""
        T = x.size(-2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rotary = Rotary(cfg.n_embd // cfg.n_head, cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int) -> None:
        super().__init__()
        self.block_size = cfg.block_size
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)  # positions come from RoPE in attention, not a learned table
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(128, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def encode_text(text: str) -> tuple[torch.Tensor, list[str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([stoi[c] for c in text], dtype=torch.long), chars


def lr_at(it: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to min_lr_frac of the peak."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    min_lr = cfg.lr * cfg.min_lr_frac
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_batch(data: torch.Tensor, cfg: Config, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[s : s + cfg.block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + cfg.block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], cfg: Config, device: str) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, cfg, device))[1].item() for _ in range(cfg.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(cfg: Config, on_eval: Callable[[dict[str, float]], None] | None = None, device: str | None = None,
          text: str | None = None) -> dict[str, Any]:
    """Train one config. on_eval(record) fires at every eval; text skips the download.

    Returns {model, chars, n_params, history, device}; each history record has
    iter, lr, train_loss, val_loss and elapsed_s.
    """
    device = device or default_device()
    torch.manual_seed(cfg.seed)

    data, chars = encode_text(text if text is not None else load_text())
    n_train = int(cfg.train_frac * len(data))
    splits = {"train": data[:n_train], "val": data[n_train:]}

    model = GPT(cfg, vocab_size=len(chars)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},  # decaying biases and norm gains only hurts
    ], lr=cfg.lr)
    history, t0 = [], time.time()

    def evaluate(it: int, lr: float) -> None:
        losses = estimate_loss(model, splits, cfg, device)
        record = {"iter": it, "lr": lr, "train_loss": losses["train"], "val_loss": losses["val"],
                  "elapsed_s": time.time() - t0}
        history.append(record)
        if on_eval is not None:
            on_eval(record)

    lr = cfg.lr
    for it in range(cfg.max_iters):
        lr = lr_at(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if it % cfg.eval_interval == 0:
            evaluate(it, lr)
        x, y = get_batch(splits["train"], cfg, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    evaluate(cfg.max_iters, lr)
    return {"model": model, "chars": chars, "n_params": n_params, "history": history, "device": device}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="exp-a-baseline", choices=sorted(PRESETS))
    args = parser.parse_args()
    cfg = resolve(PRESETS[args.preset])
    print(f"{args.preset}: {cfg}")

    def show(r: dict[str, float]) -> None:
        print(f"iter {r['iter']:5d}  lr {r['lr']:.2e}  train {r['train_loss']:.4f}  val {r['val_loss']:.4f}")

    result = train(cfg, on_eval=show)
    model, chars = result["model"], result["chars"]
    print(f"params: {result['n_params']:,}")
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=result["device"])
    print("".join(chars[i] for i in model.generate(start, 300)[0].tolist()))


if __name__ == "__main__":
    main()
