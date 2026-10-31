"""Plot validation loss for every run in runs/ and print a summary table."""
import glob
import json
import os

import matplotlib.pyplot as plt

runs = []
for path in sorted(glob.glob("runs/*.json")):
    with open(path) as f:
        r = json.load(f)
    r["name"] = os.path.splitext(os.path.basename(path))[0]
    runs.append(r)
if not runs:
    raise SystemExit("no runs found in runs/")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for r in runs:
    tok = r["config"]["batch_size"] * r["config"]["seq_len"]
    ax1.plot([s * tok / 1e6 for s in r["steps"]], r["val_loss"], label=r["name"])
    ax2.plot([t / 60 for t in r["elapsed"]], r["val_loss"], label=r["name"])
for ax, xl in [(ax1, "training tokens (M)"), (ax2, "wall-clock training time (min)")]:
    ax.set_xlabel(xl)
    ax.set_ylabel("validation loss")
    ax.grid(alpha=0.3)
    lo = min(min(r["val_loss"]) for r in runs)
    ax.set_ylim(lo - 0.1, lo + 2.0)  # zoom past the initial ~10.8 loss
ax1.set_title("Loss per token (sample efficiency)")
ax2.set_title("Loss per second (what the speedrun measures)")
ax2.legend()
fig.tight_layout()
fig.savefig("ablations.png", dpi=130)

base = next((r for r in runs if "baseline" in r["name"]), runs[0])
print(f"{'run':28s} {'val loss':>9s} {'vs base':>9s} {'minutes':>8s} {'k tok/s':>8s}")
for r in runs:
    tok = r["steps"][-1] * r["config"]["batch_size"] * r["config"]["seq_len"]
    print(f"{r['name']:28s} {r['final_val_loss']:9.4f} {r['final_val_loss']-base['final_val_loss']:+9.4f} "
          f"{r['train_time_s']/60:8.1f} {tok/r['train_time_s']/1e3:8.1f}")
print("\nsaved ablations.png")
