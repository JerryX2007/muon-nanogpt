"""
Analyze sweep results and produce, in results/ (or --out):
  lr_sweep.png    final loss vs learning rate, per optimizer, architecture and size
  curves.png      loss vs tokens and vs wall-clock time (mean over seeds, min-max band)
  advantage.png   Muon's advantage over tuned AdamW vs model size, per architecture
  results.md      summary table (paste into your README)
Works with partial results: whatever has finished gets analyzed.

    python3 analyze.py                                  # runs/sweep -> results/
    python3 analyze.py --dir runs/smoke --out results_smoke
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_RE = re.compile(r"^(lr|final)_(modern|base)_([A-Z])_(adamw|muon)_lr([0-9.e+-]+)_s(\d+)$")
COLORS = {"adamw": "tab:blue", "muon": "tab:orange"}
LABELS = {"adamw": "AdamW", "muon": "Muon"}
ARCH_LABELS = {"modern": "modern arch", "base": "baseline arch"}
ARCH_STYLE = {"modern": "o-", "base": "s--"}


def load_runs(d):
    runs = []
    for path in glob.glob(os.path.join(d, "*.json")):
        m = NAME_RE.match(os.path.splitext(os.path.basename(path))[0])
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        c = r["config"]
        r.update(phase=m[1], arch=m[2], size=m[3], opt=m[4], lr=float(m[5]), seed=int(m[6]),
                 nonembed=12 * c["n_layer"] * c["n_embd"] ** 2,
                 tokens=[s * c["batch_size"] * c["seq_len"] for s in r["steps"]])
        runs.append(r)
    return runs


def configs(runs):
    """(arch, size) pairs present, ordered by architecture then model size."""
    keys = {(r["arch"], r["size"]) for r in runs}
    size_params = {r["size"]: r["nonembed"] for r in runs}
    return sorted(keys, key=lambda k: (k[0] != "modern", size_params[k[1]]))


def mean_curve(group, xkey):
    n = min(len(r["val_loss"]) for r in group)
    x = np.mean([r[xkey][:n] for r in group], axis=0)
    ys = np.array([r["val_loss"][:n] for r in group])
    return x, ys.mean(0), ys.min(0), ys.max(0)


def reach(x, y, target):
    """First x where curve y drops to target (linear interpolation); nan if never."""
    for i in range(1, len(y)):
        if y[i] <= target:
            frac = (y[i - 1] - target) / (y[i - 1] - y[i]) if y[i - 1] != y[i] else 1.0
            return x[i - 1] + frac * (x[i] - x[i - 1])
    return float("nan")


def plot_lr_sweep(runs, out):
    lr_runs = [r for r in runs if r["phase"] == "lr"]
    if not lr_runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, opt in zip(axes, ["adamw", "muon"]):
        for arch, size in configs(lr_runs):
            pts = sorted((r["lr"], r["final_val_loss"]) for r in lr_runs
                         if r["opt"] == opt and r["arch"] == arch and r["size"] == size)
            if pts:
                ax.plot(*zip(*pts), ARCH_STYLE[arch], label=f"{ARCH_LABELS[arch]}, size {size}")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("final validation loss")
        ax.set_title(f"{LABELS[opt]}: learning-rate sweep")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lr_sweep.png"), dpi=130)
    plt.close(fig)


def plot_curves(groups, cfgs, out):
    fig, axes = plt.subplots(2, len(cfgs), figsize=(5 * len(cfgs), 8), squeeze=False)
    for col, (arch, size) in enumerate(cfgs):
        for row, (xkey, scale, xl) in enumerate([("tokens", 1e6, "training tokens (M)"),
                                                 ("elapsed", 60, "wall-clock training time (min)")]):
            ax = axes[row][col]
            lows = []
            for opt in ["adamw", "muon"]:
                g = groups.get((arch, size, opt))
                if not g:
                    continue
                x, mean, lo, hi = mean_curve(g, xkey)
                ax.plot(x / scale, mean, color=COLORS[opt], label=f"{LABELS[opt]} (lr {g[0]['lr']:g}, n={len(g)})")
                ax.fill_between(x / scale, lo, hi, color=COLORS[opt], alpha=0.2)
                lows.append(mean.min())
            if lows:
                ax.set_ylim(min(lows) - 0.05, min(lows) + 1.5)
            ax.set_xlabel(xl)
            ax.set_ylabel("validation loss")
            ax.set_title(f"{ARCH_LABELS[arch]}, size {size}: loss vs {'tokens' if row == 0 else 'time'}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=130)
    plt.close(fig)


def summarize(groups, cfgs):
    rows = []
    for arch, size in cfgs:
        a, m = groups.get((arch, size, "adamw")), groups.get((arch, size, "muon"))
        for opt, g in [("adamw", a), ("muon", m)]:
            if not g:
                continue
            finals = np.array([r["final_val_loss"] for r in g])
            rows.append(dict(
                arch=arch, size=size, nonembed=g[0]["nonembed"], opt=opt, lr=g[0]["lr"], n=len(g),
                loss=finals.mean(), std=finals.std(ddof=1) if len(g) > 1 else float("nan"),
                tps=np.mean([r["tokens_per_s"] for r in g]),
                opt_pct=100 * np.mean([r["opt_time_s"] / r["train_time_s"] for r in g]),
            ))
        if a and m:
            # what fraction of AdamW's budget does Muon need to reach AdamW's final loss?
            ax_tok, a_mean, _, _ = mean_curve(a, "tokens")
            mx_tok, m_mean, _, _ = mean_curve(m, "tokens")
            ax_t = mean_curve(a, "elapsed")[0]
            mx_t = mean_curve(m, "elapsed")[0]
            target = np.mean([r["final_val_loss"] for r in a])
            rows[-1]["tok_frac"] = reach(mx_tok, m_mean, target) / ax_tok[-1]
            rows[-1]["time_frac"] = reach(mx_t, m_mean, target) / ax_t[-1]
            rows[-1]["gap"] = rows[-2]["loss"] - rows[-1]["loss"]
    return rows


def plot_advantage(rows, out):
    pts = [r for r in rows if "gap" in r]
    if not pts:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    ax2_has_data = False
    for arch in ["modern", "base"]:
        ps = sorted((r for r in pts if r["arch"] == arch), key=lambda r: r["nonembed"])
        if not ps:
            continue
        p = [r["nonembed"] / 1e6 for r in ps]
        ax1.plot(p, [r["gap"] for r in ps], ARCH_STYLE[arch], label=ARCH_LABELS[arch])
        # "budget to match" is undefined (nan) where Muon never reached AdamW's final loss; plot only defined points
        for key, color, what in [("tok_frac", "tab:blue", "tokens"), ("time_frac", "tab:orange", "time")]:
            ok = [(x, 100 * r[key]) for x, r in zip(p, ps) if np.isfinite(r[key])]
            if ok:
                ax2_has_data = True
                ax2.plot(*zip(*ok), ARCH_STYLE[arch], color=color, label=f"{what}, {ARCH_LABELS[arch]}")
        for r, x in zip(ps, p):
            ax1.annotate(r["size"], (x, r["gap"]), textcoords="offset points", xytext=(5, 5))
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylabel("AdamW loss - Muon loss (higher = Muon better)")
    ax1.set_title("Final-loss advantage of Muon over tuned AdamW")
    ax2.axhline(100, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel("% of AdamW's budget Muon needs")
    ax2.set_title("Budget for Muon to match AdamW's final loss\n(below 100% = Muon is faster)")
    if not ax2_has_data:  # Muon never matched AdamW anywhere
        ax2.text(0.5, 0.65, "Muon did not reach AdamW's final loss\nin any configuration",
                 ha="center", va="center", transform=ax2.transAxes)
    ax1.set_xscale("log")  # shared with ax2
    for ax in (ax1, ax2):
        ax.set_xlabel("non-embedding parameters (M)")
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    if ax2_has_data:
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "advantage.png"), dpi=130)
    plt.close(fig)


def write_table(rows, out):
    def pct(v):
        return "never" if np.isnan(v) else f"{100*v:.0f}%"
    lines = ["| arch | size | non-emb params | optimizer | lr | seeds | final val loss | tok/s | optimizer time "
             "| Muon budget to match AdamW (tokens / time) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        std = "" if np.isnan(r["std"]) else f" ± {r['std']:.3f}"
        match = f"{pct(r['tok_frac'])} / {pct(r['time_frac'])}" if "tok_frac" in r else ""
        lines.append(f"| {r['arch']} | {r['size']} | {r['nonembed']/1e6:.1f}M | {LABELS[r['opt']]} | {r['lr']:g} "
                     f"| {r['n']} | {r['loss']:.4f}{std} | {r['tps']/1e3:.1f}k | {r['opt_pct']:.0f}% | {match} |")
    text = "\n".join(lines)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "runs", "sweep"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no sweep runs found in {args.dir}")
    os.makedirs(args.out, exist_ok=True)

    plot_lr_sweep(runs, args.out)
    lr_runs = [r for r in runs if r["phase"] == "lr"]
    if lr_runs:
        print("LR sweep (final val loss, 1 seed):")
        for r in sorted(lr_runs, key=lambda r: (r["arch"], r["size"], r["opt"], r["lr"])):
            print(f"  {r['arch']:6s} size {r['size']}  {LABELS[r['opt']]:5s}  lr {r['lr']:<6g}  {r['final_val_loss']:.4f}")

    groups = defaultdict(list)
    for r in runs:
        if r["phase"] == "final":
            groups[(r["arch"], r["size"], r["opt"])].append(r)
    if groups:
        cfgs = configs([r for r in runs if r["phase"] == "final"])
        plot_curves(groups, cfgs, args.out)
        rows = summarize(groups, cfgs)
        plot_advantage(rows, args.out)
        print("\nFinal comparison (mean ± std over seeds):")
        write_table(rows, args.out)
    print(f"\nfigures and table written to {args.out}/")


if __name__ == "__main__":
    main()
