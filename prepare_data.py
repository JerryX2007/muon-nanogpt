"""
Download pre-tokenized FineWeb shards (GPT-2 tokenizer) used by modded-nanogpt.

Each shard holds ~100M tokens (~200 MB). For a MacBook, 1 train shard is plenty
(the default ablation budget is far below 100M tokens).

    python prepare_data.py            # val shard + 1 train shard
    python prepare_data.py --train 3  # val shard + 3 train shards
"""
import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "kjj0/fineweb10B-gpt2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get(fname):
    if os.path.exists(os.path.join(OUT, fname)):
        print(f"already have {fname}")
        return
    hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", local_dir=OUT)
    print(f"downloaded {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1, help="number of train shards")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    get("fineweb_val_000000.bin")
    for i in range(1, args.train + 1):
        get(f"fineweb_train_{i:06d}.bin")
