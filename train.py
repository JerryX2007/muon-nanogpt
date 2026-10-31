"""
Small-scale GPT trainer for ablating modded-nanogpt techniques on a single
device (Apple Silicon MPS, CUDA, or CPU).

Every technique is a switch, so you can measure what each one adds:

    --rope      rotary position embeddings  (off = learned absolute positions)
    --qknorm    RMS-normalize queries and keys before attention
    --relu2     ReLU(x)^2 MLP activation    (off = GELU)
    --muon      Muon optimizer for hidden 2-D weights (off = AdamW everywhere)

Example:
    python train.py --name baseline
    python train.py --name full --rope --qknorm --relu2 --muon

Results (loss curves + summary) are written to runs/<name>.json.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Muon: Momentum Orthogonalized by Newton-Schulz
# --------------------------------------------------------------------------


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Approximately orthogonalize G (replace its singular values with ~1) using a
    quintic Newton-Schulz iteration. The coefficients are tuned to push singular
    values toward 1 quickly rather than to converge exactly; the result has
    singular values roughly in [0.7, 1.2], which works fine in practice.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:  # work with the wide orientation so X @ X.T is the small Gram matrix
        X = X.T
    X = X / (X.norm() + eps)  # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    SGD-momentum (Nesterov) whose update is orthogonalized before being applied.
    Only for 2-D hidden weight matrices; embeddings, the output head, and
    gains/biases should use AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so the update RMS is roughly shape-independent
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):  # x: (B, H, T, D)
        T = x.size(-2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.qknorm = cfg.qknorm
        self.rotary = Rotary(self.head_dim, cfg.seq_len) if cfg.rope else None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.qknorm:
            q, k = rms_norm(q), rms_norm(k)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.relu2 = cfg.relu2

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() if self.relu2 else F.gelu(x)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = None if cfg.rope else nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # GPT-2 style: shrink residual-branch output projections with depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets):
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(rms_norm(x)).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# --------------------------------------------------------------------------
# Data: modded-nanogpt .bin shards (256 x int32 header, then uint16 tokens)
# --------------------------------------------------------------------------


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"{path}: bad magic number"
    assert header[1] == 1, f"{path}: unsupported version"
    ntok = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntok,))


class Loader:
    """Random contiguous windows from the shards; seeded so every run sees the same batches."""

    def __init__(self, pattern, batch_size, seq_len, seed):
        files = sorted(glob.glob(pattern))
        assert files, f"no files match {pattern} -- run prepare_data.py first"
        self.shards = [load_shard(f) for f in files]
        self.B, self.T = batch_size, seq_len
        self.rng = np.random.default_rng(seed)

    def next(self, device):
        shard = self.shards[self.rng.integers(len(self.shards))]
        starts = self.rng.integers(0, len(shard) - self.T - 1, size=self.B)
        buf = np.stack([shard[s : s + self.T + 1] for s in starts]).astype(np.int64)
        buf = torch.from_numpy(buf).to(device)
        return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def lr_mult(step, total, warmup, cooldown_frac):
    """Warmup -> constant -> linear cooldown to 0 (the schedule modded-nanogpt uses)."""
    if step < warmup:
        return (step + 1) / warmup
    cooldown_start = int(total * (1 - cooldown_frac))
    if step < cooldown_start:
        return 1.0
    return max(0.0, (total - step) / (total - cooldown_start))


def build_optimizers(model, args):
    hidden_matrices = [p for n, p in model.blocks.named_parameters() if p.ndim == 2]
    other = [model.wte.weight, model.lm_head.weight] + ([model.wpe.weight] if model.wpe is not None else [])
    if args.muon:
        adam = torch.optim.AdamW(other, lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
        muon = Muon(hidden_matrices, lr=args.muon_lr, momentum=0.95, ns_steps=args.ns_steps)
        return [adam, muon]
    adam = torch.optim.AdamW(
        [{"params": hidden_matrices, "weight_decay": args.weight_decay}, {"params": other, "weight_decay": 0.0}],
        lr=args.adam_lr,
        betas=(0.9, 0.95),
    )
    return [adam]


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    losses = [model(x, y).item() for x, y in val_batches]
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    # techniques
    ap.add_argument("--rope", action="store_true")
    ap.add_argument("--qknorm", action="store_true")
    ap.add_argument("--relu2", action="store_true")
    ap.add_argument("--muon", action="store_true")
    # model size (defaults: ~11M non-embedding params, comfortable on an M4)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=50304)  # GPT-2 vocab padded to a multiple of 64
    # training budget
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tokens", type=float, default=20e6, help="total training tokens")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--cooldown_frac", type=float, default=0.3)
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz iterations in Muon")
    # eval / misc
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--val_batches", type=int, default=20, help="batches for periodic evals")
    ap.add_argument("--final_val_batches", type=int, default=100, help="batches for the final eval (less noise)")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="torch.compile (helps on CUDA; often not on MPS)")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    model = GPT(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed_params = model.wte.weight.numel() + model.lm_head.weight.numel()
    print(f"device={device}  params={n_params/1e6:.1f}M (non-embedding {(n_params-n_embed_params)/1e6:.1f}M)")
    fwd = torch.compile(model) if args.compile else model

    optimizers = build_optimizers(model, args)
    for opt in optimizers:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]

    train = Loader(os.path.join(args.data_dir, "fineweb_train_*.bin"), args.batch_size, args.seq_len, seed=args.seed)
    val = Loader(os.path.join(args.data_dir, "fineweb_val_*.bin"), args.batch_size, args.seq_len, seed=1234)
    # fixed validation set, identical for every run; periodic evals use a prefix of it
    final_val_batches = [val.next(device) for _ in range(max(args.val_batches, args.final_val_batches))]
    val_batches = final_val_batches[: args.val_batches]

    tokens_per_step = args.batch_size * args.seq_len
    total_steps = int(args.tokens // tokens_per_step)
    print(f"{total_steps} steps x {tokens_per_step} tokens = {total_steps*tokens_per_step/1e6:.1f}M tokens")

    log = {"config": vars(args), "device": device, "params": n_params, "steps": [], "val_loss": [],
           "train_steps": [], "train_loss": [], "elapsed": []}
    train_time = 0.0
    opt_time = 0.0  # time spent inside optimizer.step(), to measure Muon's overhead
    for step in range(total_steps + 1):
        last = step == total_steps
        if step % args.eval_every == 0 or last:
            vl = evaluate(fwd, val_batches)
            log["steps"].append(step)
            log["val_loss"].append(vl)
            log["elapsed"].append(train_time)
            tps = step * tokens_per_step / train_time if train_time > 0 else 0
            print(f"step {step:5d}/{total_steps}  val {vl:.4f}  time {train_time:6.1f}s  {tps/1e3:5.1f}k tok/s")
        if last:
            break

        sync(device)
        t0 = time.perf_counter()
        x, y = train.next(device)
        loss = fwd(x, y)
        loss.backward()
        sync(device)
        t1 = time.perf_counter()
        m = lr_mult(step, total_steps, args.warmup, args.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * m
            opt.step()
        model.zero_grad(set_to_none=True)
        sync(device)
        t2 = time.perf_counter()
        train_time += t2 - t0
        opt_time += t2 - t1

        if step % 10 == 0:
            log["train_steps"].append(step)
            log["train_loss"].append(loss.item())

    log["final_val_loss"] = evaluate(fwd, final_val_batches)
    log["train_time_s"] = train_time
    log["opt_time_s"] = opt_time
    log["tokens_per_s"] = total_steps * tokens_per_step / train_time
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out + ".tmp", "w") as f:  # write-then-rename so an interrupted run never leaves a half-written file
        json.dump(log, f)
    os.replace(out + ".tmp", out)
    print(f"done: final val loss {log['final_val_loss']:.4f} in {train_time/60:.1f} min "
          f"(optimizer {100*opt_time/train_time:.0f}% of time) -> {out}")


if __name__ == "__main__":
    main()
