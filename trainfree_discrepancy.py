"""Extract one label-free sentence-level discrepancy value per sample.

For each valid token, D_i is mean(abs(H_seq - H_struct)) over the hidden
dimension. The sentence score is the mean of D_i over all valid lexical tokens.
Each output row therefore contains exactly one sentence-level D value.
Human/LLM labels are read only after D has been computed, solely to route the
score to the Human or LLM output file; labels are never passed to either encoder.
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sequential import build_sequence_encoder
from structual import GraphMAEEncoder


class LabelFreeDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        adj = item["adj"]
        edge_type_mat = item["edge_type"]
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        if edge_index.size(1) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_type = torch.tensor([0], dtype=torch.long)
        else:
            src, dst = edge_index
            edge_type = edge_type_mat[src, dst].long()

        # Intentionally do not return item["label"].
        return {
            "sample_id": torch.tensor(idx, dtype=torch.long),
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "edge_index": edge_index,
            "edge_type": edge_type,
        }


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Expected a training checkpoint with model_state_dict: {path}")
    return checkpoint


def load_models(args, device):
    seq_ckpt = load_checkpoint(args.seq_ckpt, device)
    graph_ckpt = load_checkpoint(args.graph_ckpt, device)
    sa = seq_ckpt.get("args", {})
    ga = graph_ckpt.get("args", {})

    seq_model = build_sequence_encoder(
        checkpoint_args=sa,
        roberta_name=args.roberta,
        freeze_roberta=True,
    ).to(device)

    graph_model = GraphMAEEncoder(
        roberta_name=ga.get("roberta", args.roberta),
        num_relations=ga.get("num_relations", args.num_relations),
        hidden_dim=ga.get("hidden_dim", 768),
        num_layers=ga.get("num_layers", 2),
        output_dim=ga.get("output_dim", 768),
        mask_token_id=ga.get("mask_token_id", args.mask_token_id),
        freeze_roberta=True,
        dropout=ga.get("dropout", 0.1),
    ).to(device)

    seq_model.load_state_dict(seq_ckpt["model_state_dict"], strict=True)
    graph_model.load_state_dict(graph_ckpt["model_state_dict"], strict=True)
    seq_model.eval()
    graph_model.eval()
    return seq_model, graph_model


def move_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def read_label(dataset, sample_id):
    label = dataset.data[sample_id]["label"]
    return int(label.item()) if torch.is_tensor(label) else int(label)


@torch.no_grad()
def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LabelFreeDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    seq_model, graph_model = load_models(args, device)

    os.makedirs(os.path.dirname(args.human_output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.llm_output_csv) or ".", exist_ok=True)
    human_count = 0
    llm_count = 0

    with open(
        args.human_output_csv, "w", newline="", encoding="utf-8"
    ) as human_stream, open(
        args.llm_output_csv, "w", newline="", encoding="utf-8"
    ) as llm_stream:
        human_writer = csv.writer(human_stream)
        llm_writer = csv.writer(llm_stream)

        for batch in tqdm(loader, desc="Extracting sentence-level discrepancy"):
            batch = move_to_device(batch, device)
            sample_id = int(batch.pop("sample_id").item())

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_ids = input_ids.squeeze(0)

            # Inference uses the complete input, as in the original SSD model.
            # The encoders were trained by label-free masked reconstruction, but
            # no random masking is used while extracting the final token scores.
            no_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            seq_tokens = seq_model.predict_with_mask(batch, no_mask)
            struct_tokens = graph_model.predict_with_mask(batch, no_mask)

            # Same definition as new/models/models.py:
            # diff_feat = abs(h_seq - h_struct)
            # token_diff_score = diff_feat.mean(dim=-1)
            token_diff_score = torch.abs(seq_tokens - struct_tokens).mean(dim=-1)

            # Batched encoders return [batch, tokens, hidden].  Extraction
            # deliberately uses batch_size=1, so remove only that dimension
            # before indexing with the per-token validity mask.
            if token_diff_score.dim() == 2:
                if token_diff_score.size(0) != 1:
                    raise ValueError(
                        "Sentence-level extraction expects batch_size=1, got "
                        f"{token_diff_score.size(0)}"
                    )
                token_diff_score = token_diff_score.squeeze(0)
            elif token_diff_score.dim() != 1:
                raise ValueError(
                    "Unexpected token discrepancy shape: "
                    f"{tuple(token_diff_score.shape)}"
                )

            # Keep only actual lexical tokens. Excluding RoBERTa special tokens
            # and padding prevents sentence length/padding from changing D.
            valid_tokens = attention_mask.squeeze(0).bool()
            special_tokens = (token_ids == 0) | (token_ids == 1) | (token_ids == 2)
            valid_tokens = valid_tokens & (~special_tokens)

            if not valid_tokens.any():
                raise ValueError(f"Sample {sample_id} contains no valid lexical tokens")

            sentence_diff_score = token_diff_score[valid_tokens].mean().item()
            row = [sentence_diff_score]
            label = read_label(dataset, sample_id)
            if label == args.human_label:
                human_writer.writerow(row)
                human_count += 1
            elif label == args.llm_label:
                llm_writer.writerow(row)
                llm_count += 1
            else:
                raise ValueError(
                    f"Unexpected label {label} for sample {sample_id}; "
                    f"expected {args.human_label} (Human) or {args.llm_label} (LLM)"
                )

    print(f"Saved {human_count} Human samples to {args.human_output_csv}")
    print(f"Saved {llm_count} LLM samples to {args.llm_output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--human_output_csv", required=True)
    parser.add_argument("--llm_output_csv", required=True)
    parser.add_argument("--human_label", type=int, default=0)
    parser.add_argument("--llm_label", type=int, default=1)
    parser.add_argument("--seq_ckpt", default="checkpoints/sequential_encode.pt")
    parser.add_argument("--graph_ckpt", default="checkpoints/graph_encode.pt")
    parser.add_argument("--roberta", default="models/roberta-base")
    parser.add_argument("--num_relations", type=int, default=45)
    parser.add_argument("--mask_token_id", type=int, default=50264)
    parser.add_argument("--num_workers", type=int, default=0)
    extract(parser.parse_args())
