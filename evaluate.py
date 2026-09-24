import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(args):
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "downstream_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a bi-view detector checkpoint containing "
            "downstream_state_dict"
        )

    saved_args = checkpoint.get("args", {})
    view_mode = saved_args.get("view_mode", "dual")
    seq_ckpt = args.seq_ckpt or saved_args.get("seq_ckpt")
    graph_ckpt = args.graph_ckpt or saved_args.get("graph_ckpt")
    if view_mode == "dual" and (not seq_ckpt or not graph_ckpt):
        raise ValueError("Sequence and graph pretraining checkpoints are required")
    if view_mode == "sequence" and not seq_ckpt:
        raise ValueError("Sequence pretraining checkpoint is required")
    if view_mode == "structure" and not graph_ckpt:
        raise ValueError("Graph pretraining checkpoint is required")

    common_kwargs = {
        "roberta_name": saved_args.get("roberta", args.roberta),
        "num_relations": saved_args.get("num_relations", args.num_relations),
        "num_class": saved_args.get("num_class", args.num_class),
        "num_heads": saved_args.get("num_heads", args.num_heads),
        "dropout": saved_args.get("dropout", args.dropout),
    }
    if view_mode == "dual":
        model = FrozenPretrainedBiViewDetector(
            seq_ckpt=seq_ckpt,
            graph_ckpt=graph_ckpt,
            freeze_encoders=saved_args.get("freeze_encoders", True),
            **common_kwargs,
        ).to(device)
    else:
        model = FrozenPretrainedSingleViewDetector(
            view_type=view_mode,
            seq_ckpt=seq_ckpt if view_mode == "sequence" else None,
            graph_ckpt=graph_ckpt if view_mode == "structure" else None,
            **common_kwargs,
        ).to(device)
    if (
        "encoder_state_dict" in checkpoint
        and hasattr(model, "load_trainable_view_state_dict")
    ):
        model.load_trainable_view_state_dict(checkpoint["encoder_state_dict"])
    model.load_downstream_state_dict(checkpoint["downstream_state_dict"])
    model.eval()

    dataset = TextDataset(args.data_path, mmap=args.mmap_data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    labels_all = []
    scores = []
    probabilities_all = []
    discrepancy_all = []
    total_cosine = 0.0

    for batch in tqdm(loader, desc="Evaluating", ncols=100):
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
        batch_predictions = (
            probabilities[:, 1] >= args.threshold
        ).long()

        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_cosine += cosine_metric.item() * batch_size
        predictions.extend(batch_predictions.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
        scores.extend(probabilities[:, 1].cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
        discrepancy_all.extend(avg_diff.cpu().tolist())

    metrics = {
        "threshold": args.threshold,
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(labels_all, predictions),
        "precision": precision_score(
            labels_all, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels_all, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            labels_all, predictions, average="macro", zero_division=0
        ),
        "auroc": roc_auc_score(labels_all, scores)
        if len(set(labels_all)) == 2 else 0.0,
    }
    stat_key = "avg_frozen_D" if view_mode == "dual" else "avg_view_norm"
    metrics[stat_key] = sum(discrepancy_all) / max(len(discrepancy_all), 1)
    if view_mode == "dual":
        metrics["avg_frozen_abs_cosine"] = (
            total_cosine / max(total_examples, 1)
        )
    matrix = confusion_matrix(labels_all, predictions).tolist()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Confusion matrix:")
    print(matrix)

    if args.save_pred_path:
        output_dir = os.path.dirname(args.save_pred_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "preds": predictions,
            "labels": labels_all,
            "scores": scores,
            "probs": probabilities_all,
            stat_key: discrepancy_all,
            "metrics": metrics,
            "confusion_matrix": matrix,
        }, args.save_pred_path)
        print(f"Saved predictions to {args.save_pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen or end-to-end bi-view detector"
    )
    parser.add_argument(
        "--data_path", default="datasets/test.pt"
    )
    parser.add_argument(
        "--ckpt_path",
        default="checkpoints/frozen_biview_transformer_300k_final/best_auc_model.pt",
    )
    parser.add_argument(
        "--save_pred_path",
        default="",
    )
    parser.add_argument("--seq_ckpt", default=None)
    parser.add_argument("--graph_ckpt", default=None)
    parser.add_argument("--roberta", default="models/roberta-base")
    parser.add_argument("--num_relations", type=int, default=45)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
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
    evaluate(parser.parse_args())
