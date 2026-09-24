import argparse
from bisect import bisect_right
from glob import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaModel


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, mmap_data=True):
        # The C4 file also contains dense graph tensors that the sequence
        # branch never reads.  Memory mapping avoids copying all of those
        # tensors into RAM before sequence training can start.
        self.paths = sorted(glob(pt_path))
        if not self.paths:
            raise FileNotFoundError(f"No dataset files matched: {pt_path}")

        self.shards = [
            torch.load(path, mmap=mmap_data)
            for path in self.paths
        ]
        self.cumulative_sizes = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self.cumulative_sizes.append(total)

        print(
            f"Loaded {len(self.shards)} dataset shard(s), "
            f"{total:,} samples total"
        )

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        shard_idx = bisect_right(self.cumulative_sizes, idx)
        shard_start = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        item = self.shards[shard_idx][idx - shard_start]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        roberta_name,
        hidden_dim=768,
        lstm_hidden=768,
        num_layers=2,
        output_dim=768,
        mask_token_id=50264,
        freeze_roberta=True,
        dropout=0.1,
    ):
        super().__init__()

        self.mask_token_id = mask_token_id

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        roberta_dim = self.roberta.config.hidden_size
        self.freeze_roberta = freeze_roberta

        if output_dim != roberta_dim:
            raise ValueError(
                "output_dim must equal the frozen RoBERTa hidden size so that "
                "both branches reconstruct the same fixed target space."
            )

        # This projection is part of the sequence encoder, not part of the
        # reconstruction target.  The target remains the raw, frozen RoBERTa
        # representation and is therefore identical for both views.
        self.input_proj = nn.Linear(roberta_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False

    def get_roberta_hidden(self, input_ids, attention_mask):
        if self.freeze_roberta:
            with torch.no_grad():
                return self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def get_input_features(self, input_ids, attention_mask):
        return self.input_proj(
            self.get_roberta_hidden(input_ids, attention_mask)
        )

    @torch.no_grad()
    def get_reconstruction_target(self, input_ids, attention_mask):
        # Do not apply a branch-specific trainable projection here.  This raw
        # frozen representation is the common anchor for both branches.
        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.detach()

    def lstm_encode(self, x):
        residual = x
        x, _ = self.lstm(x)
        x = self.decoder(x)
        x = self.norm(x + residual)
        return x

    def build_token_mask(self, input_ids, attention_mask, mask_ratio):
        # input_ids and attention_mask are [batch_size, sequence_length].
        valid_mask = attention_mask.bool()

        special_mask = (
            (input_ids == 0) |   # <s>
            (input_ids == 2) |   # </s>
            (input_ids == 1)     # <pad>
        )

        valid_mask = valid_mask & (~special_mask)

        random_mask = torch.rand(
            input_ids.shape,
            device=input_ids.device
        ) < mask_ratio

        mask = random_mask & valid_mask

        # Match the previous batch_size=1 behavior: every non-empty example
        # contributes at least one masked token to the loss.
        empty_rows = (mask.sum(dim=1) == 0) & (valid_mask.sum(dim=1) > 0)
        for row_idx in empty_rows.nonzero(as_tuple=False).flatten():
            valid_indices = valid_mask[row_idx].nonzero(as_tuple=False).flatten()
            rand_pos = torch.randint(
                valid_indices.numel(),
                (1,),
                device=input_ids.device,
            )
            mask[row_idx, valid_indices[rand_pos]] = True

        return mask

    def predict_with_mask(self, batch, mask):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        masked_input_ids = input_ids.clone()
        masked_input_ids.masked_fill_(mask, self.mask_token_id)

        x_masked = self.get_input_features(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
        )

        x_encoded = self.lstm_encode(x_masked)
        x_pred = self.output_proj(x_encoded)

        return x_pred

    def forward(self, batch, mask_ratio=0.3, mask=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        x_target = self.get_reconstruction_target(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if mask is None:
            mask = self.build_token_mask(input_ids, attention_mask, mask_ratio)

        x_pred = self.predict_with_mask(batch, mask)

        return x_pred, x_target, mask

    @torch.no_grad()
    def encode_tokens(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        x = self.get_input_features(input_ids, attention_mask)
        x = self.lstm_encode(x).squeeze(0)

        # output_proj participates in reconstruction training, so this is no
        # longer an untrained/random projection at extraction time.
        x = self.output_proj(x)
        x = F.normalize(x, dim=-1)

        return x

    @torch.no_grad()
    def encode_sentence(self, batch):
        x = self.encode_tokens(batch)

        attention_mask = batch["attention_mask"]
        valid_mask = attention_mask.squeeze(0).bool()

        sent_emb = x[valid_mask].mean(dim=0)
        sent_emb = F.normalize(sent_emb, dim=-1)

        return sent_emb


class TransformerSequenceEncoder(nn.Module):
    """Sequence-view encoder built from Transformer self-attention blocks."""

    def __init__(
        self,
        roberta_name,
        hidden_dim=768,
        num_layers=2,
        num_heads=8,
        ffn_dim=3072,
        output_dim=768,
        mask_token_id=50264,
        freeze_roberta=True,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads: "
                f"{hidden_dim} vs {num_heads}"
            )

        self.mask_token_id = mask_token_id
        self.freeze_roberta = freeze_roberta
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        roberta_dim = self.roberta.config.hidden_size

        if output_dim != roberta_dim:
            raise ValueError(
                "output_dim must equal the frozen RoBERTa hidden size so that "
                "both branches reconstruct the same fixed target space."
            )

        self.input_proj = nn.Linear(roberta_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if freeze_roberta:
            for parameter in self.roberta.parameters():
                parameter.requires_grad = False

    def get_roberta_hidden(self, input_ids, attention_mask):
        if self.freeze_roberta:
            with torch.no_grad():
                return self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def get_input_features(self, input_ids, attention_mask):
        return self.input_proj(
            self.get_roberta_hidden(input_ids, attention_mask)
        )

    @torch.no_grad()
    def get_reconstruction_target(self, input_ids, attention_mask):
        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.detach()

    def transformer_encode(self, x, attention_mask):
        padding_mask = ~attention_mask.bool()
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )
        x = self.decoder_norm(x + self.decoder(x))
        return x

    def build_token_mask(self, input_ids, attention_mask, mask_ratio):
        valid_mask = attention_mask.bool()
        special_mask = (
            (input_ids == 0) |
            (input_ids == 2) |
            (input_ids == 1)
        )
        valid_mask = valid_mask & (~special_mask)
        random_mask = (
            torch.rand(input_ids.shape, device=input_ids.device) < mask_ratio
        )
        mask = random_mask & valid_mask

        empty_rows = (mask.sum(dim=1) == 0) & (valid_mask.sum(dim=1) > 0)
        for row_idx in empty_rows.nonzero(as_tuple=False).flatten():
            valid_indices = valid_mask[row_idx].nonzero(
                as_tuple=False
            ).flatten()
            rand_pos = torch.randint(
                valid_indices.numel(),
                (1,),
                device=input_ids.device,
            )
            mask[row_idx, valid_indices[rand_pos]] = True

        return mask

    def predict_with_mask(self, batch, mask):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        masked_input_ids = input_ids.clone()
        masked_input_ids.masked_fill_(mask, self.mask_token_id)

        x = self.get_input_features(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
        )
        x = self.transformer_encode(x, attention_mask)
        return self.output_proj(x)

    def forward(self, batch, mask_ratio=0.3, mask=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = self.get_reconstruction_target(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if mask is None:
            mask = self.build_token_mask(
                input_ids,
                attention_mask,
                mask_ratio,
            )

        prediction = self.predict_with_mask(batch, mask)
        return prediction, target, mask

    @torch.no_grad()
    def encode_tokens(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        x = self.get_input_features(input_ids, attention_mask)
        x = self.transformer_encode(x, attention_mask)
        x = self.output_proj(x).squeeze(0)
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def encode_sentence(self, batch):
        x = self.encode_tokens(batch)
        valid_mask = batch["attention_mask"].squeeze(0).bool()
        sent_emb = x[valid_mask].mean(dim=0)
        return F.normalize(sent_emb, dim=-1)


def build_sequence_encoder(
    checkpoint_args,
    roberta_name="models/roberta-base",
    freeze_roberta=True,
):
    """Build the sequence encoder recorded in a pretraining checkpoint."""
    config = dict(checkpoint_args or {})
    encoder_type = config.get("encoder_type", "lstm").lower()
    resolved_roberta = config.get("roberta", roberta_name)

    if encoder_type == "transformer":
        hidden_dim = config.get("hidden_dim", 768)
        return TransformerSequenceEncoder(
            roberta_name=resolved_roberta,
            hidden_dim=hidden_dim,
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 8),
            ffn_dim=config.get("ffn_dim", hidden_dim * 4),
            output_dim=config.get("output_dim", 768),
            mask_token_id=config.get("mask_token_id", 50264),
            freeze_roberta=freeze_roberta,
            dropout=config.get("dropout", 0.1),
        )

    if encoder_type == "lstm":
        return LSTMEncoder(
            roberta_name=resolved_roberta,
            hidden_dim=config.get("hidden_dim", 768),
            lstm_hidden=config.get("lstm_hidden", 768),
            num_layers=config.get("num_layers", 2),
            output_dim=config.get("output_dim", 768),
            mask_token_id=config.get("mask_token_id", 50264),
            freeze_roberta=freeze_roberta,
            dropout=config.get("dropout", 0.1),
        )

    raise ValueError(f"Unsupported sequence encoder type: {encoder_type}")


def reconstruction_loss(pred, target, mask, cos_weight=0.1):
    # Compute the loss in FP32 even when AMP is enabled.  Averaging each
    # sentence first preserves the weighting of the old batch_size=1 run.
    pred = pred.float()
    target = target.float()

    token_mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    token_cos = 1 - F.cosine_similarity(pred, target, dim=-1)

    token_count = mask.sum(dim=1).clamp_min(1)
    mask_float = mask.to(dtype=token_mse.dtype)

    sample_mse = (token_mse * mask_float).sum(dim=1) / token_count
    sample_cos = (token_cos * mask_float).sum(dim=1) / token_count

    valid_samples = mask.any(dim=1)
    if not valid_samples.any():
        raise RuntimeError("The batch contains no valid tokens to mask.")

    mse_loss = sample_mse[valid_samples].mean()
    cos_loss = sample_cos[valid_samples].mean()

    return mse_loss + cos_weight * cos_loss, mse_loss, cos_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    if args.init_only:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.encoder_type == "transformer":
        model = TransformerSequenceEncoder(
            roberta_name=args.roberta,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            output_dim=args.output_dim,
            mask_token_id=args.mask_token_id,
            freeze_roberta=args.freeze_roberta,
            dropout=args.dropout,
        ).to(device)
        model_name = "Transformer"
    else:
        model = LSTMEncoder(
            roberta_name=args.roberta,
            hidden_dim=args.hidden_dim,
            lstm_hidden=args.lstm_hidden,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            mask_token_id=args.mask_token_id,
            freeze_roberta=args.freeze_roberta,
            dropout=args.dropout,
        ).to(device)
        model_name = "LSTM"

    if args.init_only:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_loss": None,
                "epoch": 0,
            },
            args.save_path,
        )
        print(
            f"Saved randomly initialized {model_name} encoder to "
            f"{args.save_path} (seed={args.seed})"
        )
        return

    dataset = GraphDataset(args.data_path, mmap_data=args.mmap_data)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Freeze RoBERTa: {args.freeze_roberta}")
    print(f"Sequence model: {model_name}")
    print(f"Batch size:     {args.batch_size}")
    print(f"AMP:            {use_amp}")
    print(f"Memory-mapped:  {args.mmap_data}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")

    model.train()

    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0

        for step, batch in enumerate(loader):
            batch = {
                k: v.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                ) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                pred, target, mask = model(
                    batch,
                    mask_ratio=args.mask_ratio,
                )

            loss, mse_loss, cos_loss = reconstruction_loss(
                pred,
                target,
                mask,
                cos_weight=args.cos_weight,
            )

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cos += cos_loss.item()

            if (step + 1) % args.log_steps == 0:
                print(
                    f"[{model_name}] Epoch {epoch + 1} | "
                    f"Step {step + 1}/{len(loader)} | "
                    f"Loss {loss.item():.6f} | "
                    f"MSE {mse_loss.item():.6f} | "
                    f"Cos {cos_loss.item():.6f} | "
                    f"Masked {mask.sum().item()}"
                )

        avg_loss = total_loss / len(loader)
        avg_mse = total_mse / len(loader)
        avg_cos = total_cos / len(loader)

        print(
            f"[{model_name} Epoch {epoch + 1}] "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Avg MSE: {avg_mse:.6f} | "
            f"Avg Cos: {avg_cos:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_loss": best_loss,
                    "epoch": epoch + 1,
                },
                args.save_path,
            )

            print(
                f"Saved best {model_name} encoder to {args.save_path}, "
                f"best_loss={best_loss:.6f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="datasets/train.pt")
    parser.add_argument("--save_path", type=str, default="checkpoints/sequential_encode.pt")

    parser.add_argument("--roberta", type=str, default="models/roberta-base")

    parser.add_argument(
        "--encoder_type",
        choices=("transformer", "lstm"),
        default="transformer",
    )
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--lstm_hidden", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=3072)
    parser.add_argument("--output_dim", type=int, default=768)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--cos_weight", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--mask_token_id", type=int, default=50264)

    parser.add_argument(
        "--freeze_roberta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

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
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a reproducible randomly initialized encoder and exit.",
    )

    args = parser.parse_args()

    train(args)
