# hw1-build: caspian-7
import os
import csv
import time
import numpy as np
import torch
VOCAB_SIZE = 1024
def calculate_loss(counts, tokens):
    pairs = tokens[:-1] * VOCAB_SIZE + tokens[1:]

    probabilities = (
    (counts.reshape(-1)[pairs] + 1)
    / (
        counts.sum(axis=1)[tokens[:-1]] + VOCAB_SIZE
    )
)

    return -np.mean(np.log(probabilities))
def main():
    start_time = time.time()

    print("Loading token IDs...")

    train_ids = torch.load(
        "data/train_ids.pt",
        map_location="cpu"
    ).numpy()

    val_ids = torch.load(
        "data/val_ids.pt",
        map_location="cpu"
    ).numpy()

    print("Counting bigrams...")

    pair_indices = (
        train_ids[:-1] * VOCAB_SIZE + train_ids[1:]
    )

    counts = np.bincount(
        pair_indices,
        minlength=VOCAB_SIZE * VOCAB_SIZE
    ).reshape(VOCAB_SIZE, VOCAB_SIZE)

    train_loss = calculate_loss(counts, train_ids)
    val_loss = calculate_loss(counts, val_ids)

    elapsed = time.time() - start_time

    tokens_per_sec = len(train_ids) / elapsed
    wall_time_min = elapsed / 60.0

    print("Bigram training loss:", train_loss)
    print("Bigram validation loss:", val_loss)
    print("Wall time (minutes):", wall_time_min)

    columns = [
        "run_id",
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "max_lr",
        "max_steps",
        "ablation",
        "params",
        "best_val_loss",
        "final_train_loss",
        "tokens_per_sec",
        "wall_time_min"
    ]

    row = {
        "run_id": "bigram",
        "n_layer": 0,
        "n_head": 0,
        "n_embd": 0,
        "block_size": 0,
        "max_lr": 0,
        "max_steps": 0,
        "ablation": "bigram_add_one",
        "params": VOCAB_SIZE * VOCAB_SIZE,
        "best_val_loss": val_loss,
        "final_train_loss": train_loss,
        "tokens_per_sec": tokens_per_sec,
        "wall_time_min": wall_time_min
    }

    existing_rows = []

    if os.path.exists("runs.csv"):
        with open("runs.csv", "r", newline="") as f:
            reader = csv.DictReader(f)

            for old_row in reader:
                if old_row["run_id"] != "bigram":
                    existing_rows.append(old_row)

    existing_rows.append(row)

    with open("runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(existing_rows)

    print("Bigram results saved to runs.csv")


if __name__ == "__main__":
    main()
