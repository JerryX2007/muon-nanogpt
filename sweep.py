"""
Muon vs AdamW study, in two phases. Resumable: finished runs are skipped,
so you can Ctrl+C at any time and rerun the same command later.

  Phase 1 - learning-rate sweep (short runs, 1 seed):
      python3 sweep.py lr
  Phase 2 - final comparison (full runs, 3 seeds, best LR from phase 1):
      python3 sweep.py final
  Then:
      python3 analyze.py

Options:
  --arch modern|base  modern = RoPE + QK-Norm + ReLU^2 (default); base = the original baseline architecture
  --sizes S,M         model sizes to run (S=4x256, M=6x384, L=8x512; default S,M,L)
  --tokens 2e6        override the token budget for this phase
  --out_dir DIR       where results go (default runs/sweep); use a separate one for smoke tests
  --dry-run           print the plan without running anything

Only the optimizer and its learning rate change between AdamW and Muon runs.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "runs", "sweep")

SIZES = {  # name: (n_layer, n_head, n_embd)
    "S": (4, 4, 256),
    "M": (6, 6, 384),
    "L": (8, 8, 512),
}
ARCHS = {
    "modern": ["--rope", "--qknorm", "--relu2"],
    "base": [],
}
LR_GRID = {
    "adamw": [1e-3, 2e-3, 4e-3, 8e-3],
    "muon": [0.01, 0.02, 0.04, 0.08],
}
ADAM_LR_FOR_MUON_RUNS = 3e-3  # embeddings + output head in Muon runs (kept fixed)
PHASE_TOKENS = {"lr": 5e6, "final": 20e6}
FINAL_SEEDS = [0, 1, 2]


def run_name(phase, arch, size, opt, lr, seed):
    return f"{phase}_{arch}_{size}_{opt}_lr{lr:g}_s{seed}"


def load(out_dir, name):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def best_lr(out_dir, arch, size, opt):
    """Lowest final val loss among finished phase-1 runs for this arch/size/optimizer."""
    results = []
    for lr in LR_GRID[opt]:
        r = load(out_dir, run_name("lr", arch, size, opt, lr, 0))
        if r is not None:
            results.append((r["final_val_loss"], lr))
    if len(results) < len(LR_GRID[opt]):
        sys.exit(f"phase 1 is incomplete for {arch} / size {size} / {opt} "
                 f"({len(results)}/{len(LR_GRID[opt])} runs). Finish phase 1 first, with the same "
                 f"--arch, --sizes and --out_dir.")
    loss, lr = min(results)
    if lr in (LR_GRID[opt][0], LR_GRID[opt][-1]):
        print(f"  WARNING: best {opt} LR for {arch}/{size} ({lr:g}) is at the edge of the grid; "
              f"the true optimum may lie outside it. Consider extending LR_GRID.")
    return lr


def plan(phase, arch, sizes, out_dir):
    jobs = []
    for size in sizes:
        if phase == "lr":
            # interleave optimizers so slow drift (heat, background apps) hits both equally
            for a_lr, m_lr in zip(LR_GRID["adamw"], LR_GRID["muon"]):
                jobs.append((arch, size, "adamw", a_lr, 0))
                jobs.append((arch, size, "muon", m_lr, 0))
        else:
            a_lr, m_lr = best_lr(out_dir, arch, size, "adamw"), best_lr(out_dir, arch, size, "muon")
            for seed in FINAL_SEEDS:
                jobs.append((arch, size, "adamw", a_lr, seed))
                jobs.append((arch, size, "muon", m_lr, seed))
    return jobs


def command(phase, arch, size, opt, lr, seed, tokens, out_dir):
    n_layer, n_head, n_embd = SIZES[size]
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--name", run_name(phase, arch, size, opt, lr, seed), "--out_dir", out_dir,
           "--n_layer", str(n_layer), "--n_head", str(n_head), "--n_embd", str(n_embd),
           "--seed", str(seed), "--tokens", str(tokens), *ARCHS[arch]]
    if opt == "muon":
        cmd += ["--muon", "--muon_lr", str(lr), "--adam_lr", str(ADAM_LR_FOR_MUON_RUNS)]
    else:
        cmd += ["--adam_lr", str(lr)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["lr", "final"])
    ap.add_argument("--arch", default="modern", choices=list(ARCHS))
    ap.add_argument("--sizes", default="S,M,L")
    ap.add_argument("--tokens", type=float, default=None)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", default="", help="extra flags for train.py, e.g. --extra='--eval_every 10'")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    for s in sizes:
        if s not in SIZES:
            sys.exit(f"unknown size {s}; choose from {list(SIZES)}")
    out_dir = os.path.abspath(args.out_dir)
    tokens = args.tokens or PHASE_TOKENS[args.phase]

    jobs = plan(args.phase, args.arch, sizes, out_dir)
    done = {j for j in jobs if load(out_dir, run_name(args.phase, *j)) is not None}
    todo = [j for j in jobs if j not in done]
    print(f"phase '{args.phase}', arch '{args.arch}': {len(jobs)} runs at {tokens/1e6:g}M tokens, "
          f"{len(done)} already done  ->  {out_dir}")
    for j in jobs:
        arch, size, opt, lr, seed = j
        print(f"  [{'done' if j in done else 'todo'}] size {size}  {opt:5s}  lr {lr:<6g}  seed {seed}")
    if args.dry_run or not todo:
        return

    for i, job in enumerate(todo, 1):
        cmd = command(args.phase, *job, tokens, out_dir) + args.extra.split()
        print(f"\n=== [{i}/{len(todo)}] {run_name(args.phase, *job)} ===", flush=True)
        subprocess.run(cmd, check=True)
    nxt = "final" if args.phase == "lr" else None
    print(f"\nphase '{args.phase}' complete. Next: " +
          (f"python3 sweep.py final --arch {args.arch} --sizes {args.sizes}" if nxt else "python3 analyze.py"))


if __name__ == "__main__":
    main()
