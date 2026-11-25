# hw1-build: caspian-7
import os
import csv
import json
import time
import math
import random
import argparse
from pathlib import Path
import numpy as np
import torch
from tokenizer import BPETokenizer
from model import GPT, GPTConfig
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def prepare_data(config):
    os.makedirs("data", exist_ok=True)

    train_path = "data/train_ids.pt"
    val_path = "data/val_ids.pt"
    tokenizer_path = "tokenizer.json"

    if (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(tokenizer_path)
    ):
        print("Loading cached token IDs...")

        train_ids = torch.load(train_path, map_location="cpu")
        val_ids = torch.load(val_path, map_location="cpu")

        tokenizer = BPETokenizer.load(tokenizer_path)

        return train_ids, val_ids, tokenizer

    print("Preparing dataset...")

    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    if not data_path.is_file():
        raise FileNotFoundError(
            "Tiny Shakespeare dataset not found at: {}. "
            "Download input.txt from the Tiny Shakespeare corpus and place it "
            "in the project root (next to train.py).".format(data_path)
        )

    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    split_index = int(len(text) * 0.9)

    train_text = text[:split_index]
    val_text = text[split_index:]

    print("Training tokenizer...")

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, 1024)
    tokenizer.save(tokenizer_path)

    print("Encoding training text...")
    train_ids = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long
    )

    print("Encoding validation text...")
    val_ids = torch.tensor(
        tokenizer.encode(val_text),
        dtype=torch.long
    )

    torch.save(train_ids, train_path)
    torch.save(val_ids, val_path)

    print("Token IDs cached successfully.")

    return train_ids, val_ids, tokenizer


def get_batch(data, batch_size, block_size, device):
    max_index = len(data) - block_size - 1

    if max_index <= 0:
        raise ValueError("Dataset is too short for block_size")

    indices = torch.randint(
        0,
        max_index,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in indices
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in indices
    ])

    return x.to(device), y.to(device)


def get_lr(step, max_lr, max_steps, warmup_steps):
    min_lr = 0.1 * max_lr

    if step <= warmup_steps:
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (
        (step - warmup_steps)
        / (max_steps - warmup_steps)
    )

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, val_ids, config, device):
    model.eval()

    losses = []

    for _ in range(50):
        x, y = get_batch(
            val_ids,
            config["batch_size"],
            config["block_size"],
            device
        )

        logits, loss = model(x, y)

        losses.append(loss.item())

    average_loss = sum(losses) / len(losses)

    model.train()

    return average_loss


def save_run_result(row, run_id):
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

    existing_rows = []

    if os.path.exists("runs.csv"):
        with open("runs.csv", "r", newline="") as f:
            reader = csv.DictReader(f)

            for old_row in reader:
                if old_row["run_id"] != run_id:
                    existing_rows.append(old_row)

    existing_rows.append(row)

    with open("runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        writer.writeheader()
        writer.writerows(existing_rows)


def train(config):
    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_ids, val_ids, tokenizer = prepare_data(config)

    model_config = GPTConfig(
        vocab_size=1024,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        ablation=config.get("ablation", "none")
    )

    model = GPT(model_config).to(device)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": 0.1
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            }
        ],
        lr=config["max_lr"],
        betas=(0.9, 0.95)
    )

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log_path = "logs/{}.csv".format(config["run_id"])

    log_columns = [
        "step",
        "train_loss",
        "val_loss",
        "lr",
        "tokens_seen"
    ]

    start_time = time.time()

    tokens_seen = 0
    best_val_loss = float("inf")
    final_train_loss = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=log_columns
        )

        writer.writeheader()

        for step in range(1, config["max_steps"] + 1):

            lr = get_lr(
                step,
                config["max_lr"],
                config["max_steps"],
                config["warmup_steps"]
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = get_batch(
                train_ids,
                config["batch_size"],
                config["block_size"],
                device
            )

            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            final_train_loss = loss.item()

            tokens_seen += (
                config["batch_size"] * config["block_size"]
            )

            if step % config["eval_interval"] == 0:

                val_loss = estimate_loss(
                    model,
                    val_ids,
                    config,
                    device
                )

                print(
                    "Step: {}/{} | Train loss: {:.4f} | Val loss: {:.4f} | LR: {:.6f}".format(
                        step,
                        config["max_steps"],
                        final_train_loss,
                        val_loss,
                        lr
                    )
                )

                writer.writerow({
                    "step": step,
                    "train_loss": final_train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_seen": tokens_seen
                })

                f.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config,
                            "step": step,
                            "best_val_loss": best_val_loss
                        },
                        "checkpoints/{}_best.pt".format(
                            config["run_id"]
                        )
                    )

                    print("Best checkpoint saved.")

    wall_time = time.time() - start_time

    tokens_per_sec = tokens_seen / wall_time

    wall_time_min = wall_time / 60.0

    run_row = {
        "run_id": config["run_id"],
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "max_lr": config["max_lr"],
        "max_steps": config["max_steps"],
        "ablation": config["ablation"],
        "params": model.num_params(),
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "tokens_per_sec": tokens_per_sec,
        "wall_time_min": wall_time_min
    }

    save_run_result(run_row, config["run_id"])

    print("Training completed.")
    print("Best validation loss:", best_val_loss)
    print("Final training loss:", final_train_loss)
    print("Tokens per second:", tokens_per_sec)
    print("Wall time (minutes):", wall_time_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    train(config)


