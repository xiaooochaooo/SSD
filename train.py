import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.models import (
    FrozenPretrainedBiViewDetector,
    FrozenPretrainedSingleViewDetector,
)
from utils.data_loader import TextDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_metrics(labels, predictions, scores):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }
    metrics["auroc"] = (
        roc_auc_score(labels, scores) if len(set(labels)) == 2 else 0.0
    )
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    total_diff = 0.0
    total_cosine = 0.0
    all_predictions = []
    all_labels = []
    all_scores = []

    for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
        batch = move_batch(batch, device)
        labels = batch["label"].long().view(-1)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, _, avg_diff, cosine_metric = model(
                input_ids=batch["input_ids"],
                TPS=batch["TPS"],
                adj=batch["adj"],
                edge_type=batch["edge_type"],
                mask=batch["attention_mask"],
            )
            loss = criterion(logits.float(), labels)

        batch_size = labels.size(0)
        probabilities = F.softmax(logits.float(), dim=-1)
        predictions = logits.argmax(dim=-1)

        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_diff += avg_diff.sum().item()
        total_cosine += cosine_metric.item() * batch_size
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(probabilities[:, 1].cpu().tolist())

    metrics = compute_metrics(all_labels, all_predictions, all_scores)
    metrics["loss"] = total_loss / max(total_examples, 1)
    if hasattr(model, "view_type"):
        metrics["avg_view_norm"] = total_diff / max(total_examples, 1)
    else:
        metrics["avg_frozen_D"] = total_diff / max(total_examples, 1)
        metrics["avg_frozen_abs_cosine"] = (
            total_cosine / max(total_examples, 1)
        )
    return metrics


def checkpoint_payload(
    model,
    args,
    epoch,
    best_auroc,
    best_f1,
    metrics,
):
    payload = {
        "downstream_state_dict": model.downstream_state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_auroc": best_auroc,
        "best_f1": best_f1,
        "metrics": metrics,
        "architecture": (
            f"frozen_pretrained_{args.view_mode}_view_tps_attention"
            if args.view_mode != "dual"
            else (
                "frozen_pretrained_biview_postfusion_tps"
                if args.freeze_encoders
                else "end_to_end_biview_postfusion_tps"
            )
        ),
    }
    if args.view_mode == "dual" and not args.freeze_encoders:
        # RoBERTa stays frozen and is reloaded from its original path. Save the
        # fine-tuned Transformer/RGCN layers so evaluation exactly reproduces
        # the end-to-end model without duplicating RoBERTa in every checkpoint.
        payload["encoder_state_dict"] = model.trainable_view_state_dict()
    return payload


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_ratio,
    min_lr,
    base_lr,
):
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(total_steps - warmup_steps, 1)
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps

        progress = (current_step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    ), warmup_steps


def train(args):
    set_seed(args.seed)
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr <= args.lr:
        raise ValueError("min_lr must be in [0, lr]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Using device: {device}")

    train_dataset = TextDataset(args.train_path, mmap=args.mmap_data)
    val_dataset = TextDataset(args.val_path, mmap=args.mmap_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    if args.view_mode == "dual":
        model = FrozenPretrainedBiViewDetector(
            seq_ckpt=args.seq_ckpt,
            graph_ckpt=args.graph_ckpt,
            roberta_name=args.roberta,
            num_relations=args.num_relations,
            num_class=args.num_class,
            num_heads=args.num_heads,
            dropout=args.dropout,
            freeze_encoders=args.freeze_encoders,
        ).to(device)
        view_module = model.frozen_encoders
    else:
        if not args.freeze_encoders:
            raise ValueError(
                "Single-view ablations require frozen pretrained encoders"
            )
        model = FrozenPretrainedSingleViewDetector(
            view_type=args.view_mode,
            seq_ckpt=args.seq_ckpt if args.view_mode == "sequence" else None,
            graph_ckpt=(
                args.graph_ckpt if args.view_mode == "structure" else None
            ),
            roberta_name=args.roberta,
            num_relations=args.num_relations,
            num_class=args.num_class,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
        view_module = model.frozen_encoder
    model.train()

    frozen_parameters = sum(
        parameter.numel()
        for parameter in view_module.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if args.view_mode != "dual" or args.freeze_encoders:
        assert not any(
            parameter.requires_grad
            for parameter in view_module.parameters()
        )
    else:
        assert any(
            parameter.requires_grad
            for parameter in view_module.parameters()
        )
        assert not any(
            parameter.requires_grad
            for parameter in model.frozen_encoders.seq_encoder.roberta.parameters()
        )
        assert not any(
            parameter.requires_grad
            for parameter in model.frozen_encoders.graph_encoder.roberta.parameters()
        )
    print(f"View mode: {args.view_mode}")
    print(f"View encoders frozen: {args.freeze_encoders}")
    print(f"Total view-encoder params: {frozen_parameters:,}")
    print(f"Total trainable params: {trainable_parameters:,}")
    print(f"AMP: {use_amp}")
    print("Supervised loss: cross-entropy only")
    print("D and absolute cosine are monitoring metrics only")

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler, warmup_steps = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        base_lr=args.lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    print(f"Label smoothing: {args.label_smoothing}")
    print(
        "LR schedule: "
        f"{warmup_steps} warmup steps, cosine decay "
        f"from {args.lr:.2e} to {args.min_lr:.2e}"
    )
    print(f"Save every epoch: {args.save_every_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "training_log.jsonl")
    best_auroc = -1.0
    best_f1 = -1.0
    stat_key = (
        "avg_frozen_D" if args.view_mode == "dual" else "avg_view_norm"
    )
    stat_label = "D" if args.view_mode == "dual" else "ViewNorm"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_diff = 0.0
        total_cosine = 0.0
        correct = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=120,
        )
        for batch in progress:
            batch = move_batch(batch, device)
            labels = batch["label"].long().view(-1)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, _, avg_diff, cosine_metric = model(
                    input_ids=batch["input_ids"],
                    TPS=batch["TPS"],
                    adj=batch["adj"],
                    edge_type=batch["edge_type"],
                    mask=batch["attention_mask"],
                )
                # No decoupling term is added in either frozen or end-to-end
                # mode, so the two ablations differ only in initialization and
                # whether supervised gradients update the view encoders.
                loss = criterion(logits.float(), labels)

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=-1)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            total_diff += avg_diff.sum().item()
            total_cosine += cosine_metric.item() * batch_size
            correct += (predictions == labels).sum().item()

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total_examples:.4f}",
                stat_label: f"{avg_diff.mean().item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_metrics = {
            "loss": total_loss / total_examples,
            "accuracy": correct / total_examples,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        train_metrics[stat_key] = total_diff / total_examples
        if args.view_mode == "dual":
            train_metrics["avg_frozen_abs_cosine"] = (
                total_cosine / total_examples
            )
        val_metrics = evaluate(model, val_loader, device, use_amp)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss={train_metrics['loss']:.4f} | "
            f"Train Acc={train_metrics['accuracy']:.4f} | "
            f"LR={train_metrics['learning_rate']:.2e} | "
            f"Train {stat_label}={train_metrics[stat_key]:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Val F1={val_metrics['f1']:.4f} | "
            f"Val AUROC={val_metrics['auroc']:.4f} | "
            f"Val {stat_label}={val_metrics[stat_key]:.4f}"
        )

        is_best_auroc = val_metrics["auroc"] > best_auroc
        is_best_f1 = val_metrics["f1"] > best_f1
        if is_best_auroc:
            best_auroc = val_metrics["auroc"]
        if is_best_f1:
            best_f1 = val_metrics["f1"]

        epoch_payload = checkpoint_payload(
            model,
            args,
            epoch,
            best_auroc,
            best_f1,
            val_metrics,
        )
        if args.save_every_epoch:
            torch.save(
                epoch_payload,
                os.path.join(args.save_dir, f"epoch{epoch}.pt"),
            )

        if is_best_auroc:
            torch.save(
                epoch_payload,
                os.path.join(args.save_dir, "best_auc_model.pt"),
            )
            # Backward-compatible alias used by evaluate.py and older scripts.
            torch.save(
                epoch_payload,
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"Best model updated: AUROC={best_auroc:.6f}")

        if is_best_f1:
            torch.save(
                epoch_payload,
                os.path.join(args.save_dir, "best_f1_model.pt"),
            )
            print(f"Best F1 model updated: F1={best_f1:.6f}")

        with open(log_path, "a", encoding="utf-8") as log_stream:
            log_stream.write(json.dumps({
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }, ensure_ascii=False) + "\n")

    print(
        "Training finished. "
        f"Best validation AUROC={best_auroc:.6f}, "
        f"best validation F1={best_f1:.6f}"
    )

    if args.test_path:
        test_dataset = TextDataset(args.test_path, mmap=args.mmap_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        best_checkpoint = torch.load(
            os.path.join(args.save_dir, "best_model.pt"),
            map_location="cpu",
        )
        if (
            "encoder_state_dict" in best_checkpoint
            and hasattr(model, "load_trainable_view_state_dict")
        ):
            model.load_trainable_view_state_dict(
                best_checkpoint["encoder_state_dict"]
            )
        model.load_downstream_state_dict(
            best_checkpoint["downstream_state_dict"]
        )
        test_metrics = evaluate(model, test_loader, device, use_amp)
        print("Test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the frozen label-free bi-view detector"
    )
    parser.add_argument("--train_path", default="datasets/train.pt")
    parser.add_argument("--val_path", default="datasets/val.pt")
    parser.add_argument(
        "--test_path", default=""
    )
    parser.add_argument(
        "--seq_ckpt",
        default="checkpoints/C4/sequential_transformer_c4_300k.pt",
    )
    parser.add_argument(
        "--graph_ckpt",
        default="checkpoints/C4/graph_rgcn_c4_300k.pt",
    )
    parser.add_argument("--roberta", default="models/roberta-base")
    parser.add_argument("--num_relations", type=int, default=45)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument(
        "--view_mode",
        choices=("dual", "sequence", "structure"),
        default="dual",
        help="Use both pretrained views or one frozen pretrained view.",
    )
    parser.add_argument(
        "--freeze_encoders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the sequence/structure encoders. Use "
            "--no-freeze_encoders for supervised end-to-end training; "
            "RoBERTa remains frozen."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="checkpoints/frozen_biview")
    parser.add_argument(
        "--save_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mmap_data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train(parser.parse_args())
