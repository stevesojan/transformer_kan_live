import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


UNK_TOKEN = "<UNK>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> Tuple[str, str, str]:
    """
    Tiny Shakespeare fixed split:
    - Train: first 30,000 lines (75%)
    - Val: lines 30,000-31,999 (5%)
    - Test: lines 32,000-39,999 (20%)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 40_000:
        raise ValueError(
            f"Expected at least 40,000 lines, found {len(lines)} in {file_path}."
        )

    lines = lines[:40_000]
    train_text = "".join(lines[:30_000])
    val_text = "".join(lines[30_000:32_000])
    test_text = "".join(lines[32_000:40_000])
    return train_text, val_text, test_text


def build_vocab(train_text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(train_text))
    if UNK_TOKEN in chars:
        raise ValueError(f"Special token {UNK_TOKEN} collides with data vocabulary.")

    stoi = {ch: i for i, ch in enumerate(chars)}
    stoi[UNK_TOKEN] = len(stoi)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_data(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    unk_idx = stoi[UNK_TOKEN]
    encoded = [stoi.get(ch, unk_idx) for ch in text]
    return torch.tensor(encoded, dtype=torch.long)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KAN-only next-char prediction:
    input:  x[t:t+block_size]
    target: x[t+block_size] (single next token)
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Data length {len(data)} must be greater than block_size {block_size}."
        )

    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + block_size] for i in starts])
    return x.to(device), y.to(device)


class KANOnlyTextModel(nn.Module):
    """
    Character window -> embedding -> flatten -> KAN -> next-token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        device: torch.device,
        d_model: int = 128,
        kan_hidden_dim: int = 128,
        dropout: float = 0.1,
        kan_grid: int = 3,
        kan_k: int = 3,
        kan_seed: int = 42,
        kan_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device_ref = device
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.kan_chunk_size = kan_chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        try:
            from kan import KAN  # type: ignore
        except ImportError:
            import sys

            script_dir = Path(__file__).resolve().parent
            pykan_root = script_dir / "pykan"
            if pykan_root.exists():
                sys.path.append(str(pykan_root))
                from kan import KAN  # type: ignore
            else:
                raise ImportError(
                    "Could not import KAN. Install pykan or place local repo at ./pykan."
                )

        in_dim = block_size * d_model
        self.kan = KAN(
            width=[in_dim, kan_hidden_dim, vocab_size],
            grid=kan_grid,
            k=kan_k,
            base_fun="silu",
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
            seed=kan_seed,
        )
        self.kan.speed()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        bsz, seq_len = idx.shape
        if seq_len != self.block_size:
            raise ValueError(
                f"Expected sequence length {self.block_size}, got {seq_len}."
            )

        x = self.token_embedding(idx)  # (B, T, C)
        x = self.embedding_dropout(x)
        x = x.contiguous().reshape(bsz, seq_len * self.d_model)  # (B, T*C)

        if self.kan_chunk_size <= 0 or x.size(0) <= self.kan_chunk_size:
            logits = self.kan(x)
        else:
            logits = torch.empty((x.size(0), self.vocab_size), device=x.device, dtype=x.dtype)
            for i in range(0, x.size(0), self.kan_chunk_size):
                j = i + self.kan_chunk_size
                logits[i:j] = self.kan(x[i:j])

        return logits  # (B, vocab_size)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class EpochStats:
    loss: float
    epoch_time_sec: float
    tokens_per_sec: float
    global_step: int


def train_one_epoch(
    model: nn.Module,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    block_size: int,
    batch_size: int,
    grad_clip: float,
    steps_per_epoch: int,
    global_step_start: int,
    log_every: int,
) -> EpochStats:
    model.train()
    start_time = time.perf_counter()
    step_start = time.perf_counter()

    running_loss = 0.0
    tokens_seen = 0
    global_step = global_step_start

    for step in range(steps_per_epoch):
        x, y = get_batch(train_data, block_size, batch_size, device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        tokens_seen += x.numel()
        global_step += 1

        if (step % log_every == 0) or (step == steps_per_epoch - 1):
            elapsed = time.perf_counter() - step_start
            avg_step = elapsed / max(step + 1, 1)
            print(
                f"  step {step:04d}/{steps_per_epoch} | "
                f"loss={loss.item():.4f} | "
                f"avg_step_time={avg_step:.3f}s",
                flush=True,
            )

    epoch_time = time.perf_counter() - start_time
    avg_loss = running_loss / steps_per_epoch
    tokens_per_sec = tokens_seen / max(epoch_time, 1e-9)
    return EpochStats(
        loss=avg_loss,
        epoch_time_sec=epoch_time,
        tokens_per_sec=tokens_per_sec,
        global_step=global_step,
    )


def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    block_size: int,
    batch_size: int,
) -> float:
    """
    Deterministic full-split evaluation using non-overlapping windows.
    """
    model.eval()
    starts = list(range(0, len(data) - block_size - 1, block_size))
    if not starts:
        raise ValueError(
            f"Evaluation split too small for block_size={block_size}. len(data)={len(data)}"
        )

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk_starts = starts[i : i + batch_size]
            x = torch.stack([data[s : s + block_size] for s in chunk_starts]).to(device)
            y = torch.stack([data[s + block_size] for s in chunk_starts]).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def compute_metrics(
    final_train_loss: float,
    final_val_loss: float,
    final_test_loss: float,
    best_val_epoch: int,
    perplexity_threshold_step: Optional[int],
    epoch_times: List[float],
    epoch_tps: List[float],
    total_params: int,
    batch_size: int,
    block_size: int,
) -> Dict[str, float]:
    val_perplexity = math.exp(final_val_loss)
    test_perplexity = math.exp(final_test_loss)
    generalization_gap = final_train_loss - final_val_loss

    flops_per_forward = 2 * total_params * block_size
    flops_per_batch = flops_per_forward * batch_size

    return {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_test_loss": final_test_loss,
        "val_perplexity": val_perplexity,
        "test_perplexity": test_perplexity,
        "generalization_gap": generalization_gap,
        "best_val_epoch": float(best_val_epoch),
        "perplexity_threshold_step": (
            float(perplexity_threshold_step)
            if perplexity_threshold_step is not None
            else float("nan")
        ),
        "avg_epoch_time_sec": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        "avg_tokens_per_sec": float(np.mean(epoch_tps)) if epoch_tps else float("nan"),
        "estimated_flops_per_batch": float(flops_per_batch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Evaluate KAN-only autoregressive model on Tiny Shakespeare."
    )
    parser.add_argument("--data_path", type=str, default="tiny_shakespeare.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ppl_threshold", type=float, default=50.0)
    parser.add_argument("--log_every", type=int, default=50)

    # Match KAN configs from transformer-decoder-kan.py
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--kan_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kan_grid", type=int, default=3)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_seed", type=int, default=42)
    parser.add_argument("--kan_chunk_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 0:
        torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Loading dataset...")
    train_text, val_text, test_text = load_data(args.data_path)

    print("Building vocabulary from training split only...")
    stoi, itos = build_vocab(train_text)
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    print("Encoding splits...")
    train_data = encode_data(train_text, stoi)
    val_data = encode_data(val_text, stoi)
    test_data = encode_data(test_text, stoi)
    del itos

    model = KANOnlyTextModel(
        vocab_size=vocab_size,
        block_size=args.block_size,
        device=device,
        d_model=args.d_model,
        kan_hidden_dim=args.kan_hidden_dim,
        dropout=args.dropout,
        kan_grid=args.kan_grid,
        kan_k=args.kan_k,
        kan_seed=args.kan_seed,
        kan_chunk_size=args.kan_chunk_size,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_val_epoch = -1
    threshold_step: Optional[int] = None
    global_step = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    epoch_times: List[float] = []
    epoch_tps: List[float] = []

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_stats = train_one_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            block_size=args.block_size,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            steps_per_epoch=args.steps_per_epoch,
            global_step_start=global_step,
            log_every=max(1, args.log_every),
        )
        global_step = epoch_stats.global_step

        val_loss = evaluate(
            model=model,
            data=val_data,
            device=device,
            block_size=args.block_size,
            batch_size=args.batch_size,
        )
        val_ppl = math.exp(val_loss)

        train_losses.append(epoch_stats.loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_stats.epoch_time_sec)
        epoch_tps.append(epoch_stats.tokens_per_sec)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
        if threshold_step is None and val_ppl < args.ppl_threshold:
            threshold_step = global_step

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={epoch_stats.loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.4f} | "
            f"time={epoch_stats.epoch_time_sec:.2f}s | "
            f"tokens/s={epoch_stats.tokens_per_sec:.2f}"
        )

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_test_loss = evaluate(
        model=model,
        data=test_data,
        device=device,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    metrics = compute_metrics(
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_test_loss=final_test_loss,
        best_val_epoch=best_val_epoch,
        perplexity_threshold_step=threshold_step,
        epoch_times=epoch_times,
        epoch_tps=epoch_tps,
        total_params=total_params,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    threshold_step_str = (
        str(int(metrics["perplexity_threshold_step"]))
        if not math.isnan(metrics["perplexity_threshold_step"])
        else "Not reached"
    )

    print("\n=================================================")
    print("MODEL: KAN-only autoregressive text model")
    print("=================================================")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"Final Training Loss: {metrics['final_train_loss']:.6f}")
    print(f"Final Validation Loss: {metrics['final_val_loss']:.6f}")
    print(f"Final Test Loss: {metrics['final_test_loss']:.6f}\n")
    print(f"Validation Perplexity: {metrics['val_perplexity']:.6f}")
    print(f"Test Perplexity: {metrics['test_perplexity']:.6f}\n")
    print(f"Generalization Gap: {metrics['generalization_gap']:.6f}\n")
    print(f"Best Validation Epoch: {int(metrics['best_val_epoch'])}")
    print(f"Perplexity Threshold Step: {threshold_step_str}\n")
    print(f"Training Time Per Epoch: {metrics['avg_epoch_time_sec']:.4f} sec")
    print(f"Tokens per Second: {metrics['avg_tokens_per_sec']:.4f}")
    print(f"Estimated FLOPs per Batch: {metrics['estimated_flops_per_batch']:.4e}")
    print("=================================================")


if __name__ == "__main__":
    main()
