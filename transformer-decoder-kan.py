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
    Load Tiny Shakespeare and perform deterministic line split:
    - Train: first 30,000 lines
    - Val: lines 30,000-31,999
    - Test: lines 32,000-39,999
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 40_000:
        raise ValueError(
            f"Expected at least 40,000 lines, found {len(lines)} in {file_path}."
        )

    # Enforce exact 40,000-line protocol for reproducibility.
    lines = lines[:40_000]
    train_lines = lines[:30_000]
    val_lines = lines[30_000:32_000]
    test_lines = lines[32_000:40_000]

    train_text = "".join(train_lines)
    val_text = "".join(val_lines)
    test_text = "".join(test_lines)
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
    if len(data) <= block_size:
        raise ValueError(
            f"Data length {len(data)} must be greater than block_size {block_size}."
        )

    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True values are masked in MultiheadAttention when bool mask is used.
    return torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1
    )


class KANFeedForward(nn.Module):
    """
    KAN replacement for the standard FFN block.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        kan_grid: int,
        kan_k: int,
        dropout: float,
        kan_seed: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.chunk_size = chunk_size

        try:
            from kan import KAN  # type: ignore
        except ImportError:
            # Fallback to local pykan repo in the same workspace.
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

        self.kan = KAN(
            width=[d_model, hidden_dim, d_model],
            grid=kan_grid,
            k=kan_k,
            base_fun="silu",
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
            seed=kan_seed,
        )
        # Disable symbolic branch for training speed.
        self.kan.speed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> flatten to (B*T, C) for KAN.
        bsz, seq_len, d_model = x.shape
        # Force contiguous memory layout before flattening to reduce hidden copies.
        x = x.contiguous()
        flat = x.reshape(-1, d_model)
        # Chunk KAN calls to improve cache locality on CPU.
        if self.chunk_size <= 0 or flat.size(0) <= self.chunk_size:
            out = self.kan(flat)
        else:
            out = torch.empty_like(flat)
            for i in range(0, flat.size(0), self.chunk_size):
                j = i + self.chunk_size
                out[i:j] = self.kan(flat[i:j])
        out = out.reshape(bsz, seq_len, d_model)
        out = self.dropout(out)
        return out


class TransformerDecoderLayerKAN(nn.Module):
    """
    Decoder-only transformer layer that uses self-attention + KAN FFN.
    Compatible with nn.TransformerDecoder.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        kan_hidden_dim: int,
        dropout: float,
        kan_grid: int,
        kan_k: int,
        kan_seed: int,
        ff_chunk_size: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.kan_ff = KANFeedForward(
            d_model=d_model,
            hidden_dim=kan_hidden_dim,
            kan_grid=kan_grid,
            kan_k=kan_k,
            dropout=dropout,
            kan_seed=kan_seed,
            chunk_size=ff_chunk_size,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        del memory, memory_mask, memory_key_padding_mask, memory_is_causal

        x = tgt
        attn_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=tgt_is_causal,
        )[0]
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.kan_ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class TransformerDecoderKAN(nn.Module):
    """
    Transformer Decoder - KAN architecture.
    Constructor matches required placeholder signature:
      (vocab_size, block_size, device, ...)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        device: torch.device,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        kan_hidden_dim: int = 256,
        dropout: float = 0.1,
        kan_grid: int = 5,
        kan_k: int = 3,
        kan_seed: int = 42,
        ff_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device_ref = device
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        layer = TransformerDecoderLayerKAN(
            d_model=d_model,
            nhead=nhead,
            kan_hidden_dim=kan_hidden_dim,
            dropout=dropout,
            kan_grid=kan_grid,
            kan_k=kan_k,
            kan_seed=kan_seed,
            ff_chunk_size=ff_chunk_size,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds block_size {self.block_size}."
            )

        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.embedding_dropout(x)

        causal_mask = build_causal_mask(seq_len, idx.device)

        # Decoder API requires memory, but this is decoder-only language modeling.
        dummy_memory = torch.zeros(
            (bsz, 1, self.d_model), device=idx.device, dtype=x.dtype
        )

        x = self.decoder(
            tgt=x,
            memory=dummy_memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


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

    running_loss = 0.0
    global_step = global_step_start
    tokens_seen = 0

    step_start = time.perf_counter()
    for step in range(steps_per_epoch):
        x, y = get_batch(train_data, block_size, batch_size, device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        tokens_seen += x.numel()
        global_step += 1

        if (step % log_every == 0) or (step == steps_per_epoch - 1):
            step_time = time.perf_counter() - step_start
            avg_step_time = step_time / max(step + 1, 1)
            print(
                f"  step {step:04d}/{steps_per_epoch} | "
                f"loss={loss.item():.4f} | "
                f"avg_step_time={avg_step_time:.3f}s",
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
    Deterministic full-split evaluation with teacher forcing.
    Uses non-overlapping windows to cover data efficiently.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    starts = list(range(0, len(data) - block_size - 1, block_size))
    if not starts:
        raise ValueError(
            f"Evaluation split too small for block_size={block_size}. "
            f"len(data)={len(data)}"
        )

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk_starts = starts[i : i + batch_size]
            x = torch.stack([data[s : s + block_size] for s in chunk_starts]).to(device)
            y = torch.stack(
                [data[s + 1 : s + block_size + 1] for s in chunk_starts]
            ).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return total_loss / max(total_tokens, 1)


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

    metrics = {
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
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Evaluate Transformer Decoder - KAN on Tiny Shakespeare."
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

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kan_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kan_grid", type=int, default=3)  # grid
    parser.add_argument("--kan_k", type=int, default=3)  # spline
    parser.add_argument("--kan_seed", type=int, default=42)
    parser.add_argument("--ff_chunk_size", type=int, default=512)

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
    del itos  # mapping retained via stoi; decode not required for this pipeline.

    # Placeholder model class for easy swapping with any compatible model.
    model_class = TransformerDecoderKAN
    model = model_class(
        vocab_size=vocab_size,
        block_size=args.block_size,
        device=device,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        kan_hidden_dim=args.kan_hidden_dim,
        dropout=args.dropout,
        kan_grid=args.kan_grid,
        kan_k=args.kan_k,
        kan_seed=args.kan_seed,
        ff_chunk_size=args.ff_chunk_size,
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

    print("Evaluating on test split...")
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
    print("MODEL: Transformer Decoder - KAN architecture")
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
