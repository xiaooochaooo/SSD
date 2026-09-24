import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from sequential import build_sequence_encoder
from structual import GraphMAEEncoder


def _checkpoint_payload(path):
    checkpoint = torch.load(path, map_location="cpu", mmap=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Expected a pretraining checkpoint with model_state_dict: {path}"
        )
    return checkpoint


class FrozenPretrainedBiViewEncoder(nn.Module):
    """Sequence and structure encoders with optional end-to-end fine-tuning."""

    def __init__(
        self,
        seq_ckpt,
        graph_ckpt,
        roberta_name="models/roberta-base",
        num_relations=45,
        freeze_encoders=True,
    ):
        super().__init__()
        self.freeze_encoders = freeze_encoders

        seq_payload = _checkpoint_payload(seq_ckpt)
        graph_payload = _checkpoint_payload(graph_ckpt)
        seq_args = seq_payload.get("args", {})
        graph_args = graph_payload.get("args", {})

        self.seq_encoder = build_sequence_encoder(
            checkpoint_args=seq_args,
            roberta_name=roberta_name,
            freeze_roberta=True,
        )
        self.graph_encoder = GraphMAEEncoder(
            roberta_name=graph_args.get("roberta", roberta_name),
            num_relations=graph_args.get("num_relations", num_relations),
            hidden_dim=graph_args.get("hidden_dim", 768),
            num_layers=graph_args.get("num_layers", 2),
            output_dim=graph_args.get("output_dim", 768),
            mask_token_id=graph_args.get("mask_token_id", 50264),
            freeze_roberta=True,
            dropout=graph_args.get("dropout", 0.1),
        )

        self.seq_encoder.load_state_dict(
            seq_payload["model_state_dict"], strict=True
        )
        self.graph_encoder.load_state_dict(
            graph_payload["model_state_dict"], strict=True
        )

        seq_dim = seq_args.get("output_dim", 768)
        graph_dim = graph_args.get("output_dim", 768)
        if seq_dim != graph_dim:
            raise ValueError(
                "The two frozen encoders must share an output space: "
                f"sequence={seq_dim}, structure={graph_dim}"
            )
        self.output_dim = seq_dim

        if self.freeze_encoders:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.eval()
        else:
            for parameter in self.seq_encoder.parameters():
                parameter.requires_grad = True
            for parameter in self.graph_encoder.parameters():
                parameter.requires_grad = True
            # RoBERTa is kept frozen in every ablation so that the only changed
            # factor is view-encoder pretraining/fine-tuning.
            for parameter in self.seq_encoder.roberta.parameters():
                parameter.requires_grad = False
            for parameter in self.graph_encoder.roberta.parameters():
                parameter.requires_grad = False

        del seq_payload
        del graph_payload

    def train(self, mode=True):
        if self.freeze_encoders:
            # Frozen encoders stay deterministic while downstream layers train.
            super().train(False)
        else:
            super().train(mode)
            # Frozen RoBERTa remains deterministic even when the view-specific
            # Transformer/RGCN layers are fine-tuned.
            self.seq_encoder.roberta.eval()
            self.graph_encoder.roberta.eval()
        return self

    @staticmethod
    def dense_graph_to_sparse(adj, edge_type):
        if adj.dim() != 3:
            raise ValueError(f"Expected adj [B,L,L], got {tuple(adj.shape)}")

        batch_size, seq_len, _ = adj.shape
        graph_idx, src, dst = adj.ne(0).nonzero(as_tuple=True)

        if graph_idx.numel() == 0:
            node_ids = torch.arange(
                batch_size * seq_len,
                device=adj.device,
                dtype=torch.long,
            )
            edge_index = torch.stack([node_ids, node_ids], dim=0)
            sparse_edge_type = torch.zeros_like(node_ids)
        else:
            offset = graph_idx * seq_len
            edge_index = torch.stack([offset + src, offset + dst], dim=0)
            if edge_type is None:
                sparse_edge_type = torch.zeros(
                    edge_index.size(1),
                    dtype=torch.long,
                    device=adj.device,
                )
            else:
                sparse_edge_type = edge_type[graph_idx, src, dst].long()

        return edge_index, sparse_edge_type

    def forward(self, input_ids, attention_mask, adj, edge_type):
        if self.freeze_encoders:
            self.eval()
        edge_index, sparse_edge_type = self.dense_graph_to_sparse(
            adj, edge_type
        )
        encoder_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "edge_index": edge_index,
            "edge_type": sparse_edge_type,
        }
        no_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # predict_with_mask(no_mask) is exactly the representation used by
        # trainfree_discrepancy.py, so the classifier and the reported D share
        # the same frozen feature definition.
        seq_tokens = self.seq_encoder.predict_with_mask(
            encoder_batch, no_mask
        )
        graph_tokens = self.graph_encoder.predict_with_mask(
            encoder_batch, no_mask
        )

        if self.freeze_encoders:
            return seq_tokens.detach(), graph_tokens.detach()
        return seq_tokens, graph_tokens

    @staticmethod
    def _view_state_without_roberta(module):
        return {
            key: value
            for key, value in module.state_dict().items()
            if not key.startswith("roberta.")
        }

    def trainable_view_state_dict(self):
        """Save fine-tuned view layers without duplicating frozen RoBERTa."""
        return {
            "seq_encoder": self._view_state_without_roberta(self.seq_encoder),
            "graph_encoder": self._view_state_without_roberta(self.graph_encoder),
        }

    @staticmethod
    def _load_view_state_without_roberta(module, state_dict):
        current_state = module.state_dict()
        current_state.update(state_dict)
        module.load_state_dict(current_state, strict=True)

    def load_trainable_view_state_dict(self, state_dict):
        self._load_view_state_without_roberta(
            self.seq_encoder,
            state_dict["seq_encoder"],
        )
        self._load_view_state_without_roberta(
            self.graph_encoder,
            state_dict["graph_encoder"],
        )


class FrozenStructureSemanticDiscrepancyAttention(nn.Module):
    """Original discrepancy-gated fusion with token TPS added post-fusion."""

    def __init__(self, encoder_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = encoder_dim
        self.num_heads = num_heads
        self.combined_dim = encoder_dim * 2
        if self.combined_dim % num_heads != 0:
            raise ValueError("2 * encoder_dim must be divisible by num_heads")
        self.head_dim = self.combined_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.gate_net = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, 1),
            nn.Sigmoid(),
        )

        # enhanced_feat and fixed diff_feat are fused first. TPS is appended
        # afterwards, immediately before the unchanged Q/K/V attention stage.
        self.post_tps_dim = self.combined_dim + 1
        self.layer_norm = nn.LayerNorm(self.post_tps_dim)
        self.W_q = nn.Linear(self.post_tps_dim, self.combined_dim)
        self.W_k = nn.Linear(self.post_tps_dim, self.combined_dim)
        self.W_v = nn.Linear(self.post_tps_dim, self.combined_dim)
        self.out_proj = nn.Linear(self.combined_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_seq,
        x_struct,
        TPS,
        mask=None,
        discrepancy_mask=None,
    ):
        if x_seq.shape != x_struct.shape:
            raise ValueError(
                "Sequence and structure features must have identical shapes, "
                f"got {tuple(x_seq.shape)} and {tuple(x_struct.shape)}"
            )

        batch_size, seq_len, hidden_dim = x_seq.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected encoder dim {self.hidden_dim}, got {hidden_dim}"
            )
        if TPS.shape != x_seq.shape[:2]:
            raise ValueError(
                f"Expected TPS [B,L]={tuple(x_seq.shape[:2])}, "
                f"got {tuple(TPS.shape)}"
            )

        # No trainable projection is applied before subtraction: this is the
        # fixed, label-free D learned during common-target pretraining.
        diff_feat = torch.abs(x_seq - x_struct)
        token_diff = diff_feat.float().mean(dim=-1)

        metric_mask = discrepancy_mask if discrepancy_mask is not None else mask
        if metric_mask is not None:
            metric_mask_float = metric_mask.to(token_diff.dtype)
            avg_diff = (
                (token_diff * metric_mask_float).sum(dim=1)
                / metric_mask_float.sum(dim=1).clamp_min(1.0)
            )
            cosine = torch.abs(F.cosine_similarity(
                x_seq.float(), x_struct.float(), dim=-1
            ))
            decouple_metric = (
                (cosine * metric_mask_float).sum()
                / metric_mask_float.sum().clamp_min(1.0)
            )
        else:
            avg_diff = token_diff.mean(dim=1)
            decouple_metric = torch.abs(F.cosine_similarity(
                x_seq.float(), x_struct.float(), dim=-1
            )).mean()

        gate_input = torch.cat([x_struct, diff_feat], dim=-1)
        alpha = self.gate_net(gate_input)
        enhanced_feat = (1 - alpha) * x_struct + alpha * x_seq

        combined_feat = torch.cat([enhanced_feat, diff_feat], dim=-1)
        tps_feature = TPS.unsqueeze(-1).to(combined_feat.dtype)
        combined_feat = torch.cat([combined_feat, tps_feature], dim=-1)
        combined_feat = self.layer_norm(combined_feat)

        Q = self.W_q(combined_feat).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        K = self.W_k(combined_feat).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        V = self.W_v(combined_feat).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1).unsqueeze(2)
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(~key_mask, mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.combined_dim
        )
        out = self.out_proj(out)

        if mask is not None:
            mask_value = torch.finfo(out.dtype).min
            out = out.masked_fill(~mask.bool().unsqueeze(-1), mask_value)
        sent_vec, _ = out.max(dim=1)

        # decouple_metric is diagnostic only; it is intentionally detached and
        # must not be added to the supervised objective.
        return (
            sent_vec,
            attn_weights,
            avg_diff.detach(),
            decouple_metric.detach(),
        )


class FrozenPretrainedBiViewDetector(nn.Module):
    def __init__(
        self,
        seq_ckpt,
        graph_ckpt,
        roberta_name="models/roberta-base",
        num_relations=45,
        num_class=2,
        num_heads=4,
        dropout=0.6,
        freeze_encoders=True,
    ):
        super().__init__()
        self.freeze_encoders = freeze_encoders
        self.frozen_encoders = FrozenPretrainedBiViewEncoder(
            seq_ckpt=seq_ckpt,
            graph_ckpt=graph_ckpt,
            roberta_name=roberta_name,
            num_relations=num_relations,
            freeze_encoders=freeze_encoders,
        )
        encoder_dim = self.frozen_encoders.output_dim
        self.attention = FrozenStructureSemanticDiscrepancyAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(encoder_dim, num_class)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoders:
            self.frozen_encoders.eval()
        return self

    def forward(self, input_ids, TPS, adj, edge_type=None, mask=None):
        if mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)

        x_seq, x_struct = self.frozen_encoders(
            input_ids=input_ids,
            attention_mask=mask,
            adj=adj,
            edge_type=edge_type,
        )

        lexical_mask = mask.bool()
        lexical_mask = lexical_mask & input_ids.ne(0)
        lexical_mask = lexical_mask & input_ids.ne(1)
        lexical_mask = lexical_mask & input_ids.ne(2)

        sent_vec, attn_weights, avg_diff, decouple_metric = self.attention(
            x_seq=x_seq,
            x_struct=x_struct,
            TPS=TPS,
            mask=mask,
            discrepancy_mask=lexical_mask,
        )
        logits = self.fc(sent_vec)
        return logits, attn_weights, avg_diff, decouple_metric

    def downstream_state_dict(self):
        return {
            "attention": self.attention.state_dict(),
            "fc": self.fc.state_dict(),
        }

    def load_downstream_state_dict(self, state_dict):
        self.attention.load_state_dict(state_dict["attention"], strict=True)
        self.fc.load_state_dict(state_dict["fc"], strict=True)
    def trainable_view_state_dict(self):
        return self.frozen_encoders.trainable_view_state_dict()

    def load_trainable_view_state_dict(self, state_dict):
        self.frozen_encoders.load_trainable_view_state_dict(state_dict)


class FrozenPretrainedSingleViewEncoder(nn.Module):
    """One frozen pretrained sequence or structure encoder."""

    def __init__(
        self,
        view_type,
        seq_ckpt=None,
        graph_ckpt=None,
        roberta_name="models/roberta-base",
        num_relations=45,
    ):
        super().__init__()
        if view_type not in {"sequence", "structure"}:
            raise ValueError(f"Unsupported single-view type: {view_type}")
        self.view_type = view_type

        if view_type == "sequence":
            if not seq_ckpt:
                raise ValueError("seq_ckpt is required for sequence-only mode")
            payload = _checkpoint_payload(seq_ckpt)
            checkpoint_args = payload.get("args", {})
            self.encoder = build_sequence_encoder(
                checkpoint_args=checkpoint_args,
                roberta_name=roberta_name,
                freeze_roberta=True,
            )
        else:
            if not graph_ckpt:
                raise ValueError("graph_ckpt is required for structure-only mode")
            payload = _checkpoint_payload(graph_ckpt)
            checkpoint_args = payload.get("args", {})
            self.encoder = GraphMAEEncoder(
                roberta_name=checkpoint_args.get("roberta", roberta_name),
                num_relations=checkpoint_args.get("num_relations", num_relations),
                hidden_dim=checkpoint_args.get("hidden_dim", 768),
                num_layers=checkpoint_args.get("num_layers", 2),
                output_dim=checkpoint_args.get("output_dim", 768),
                mask_token_id=checkpoint_args.get("mask_token_id", 50264),
                freeze_roberta=True,
                dropout=checkpoint_args.get("dropout", 0.1),
            )

        self.encoder.load_state_dict(payload["model_state_dict"], strict=True)
        self.output_dim = checkpoint_args.get("output_dim", 768)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()
        del payload

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, adj=None, edge_type=None):
        self.eval()
        encoder_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.view_type == "structure":
            if adj is None:
                raise ValueError("adj is required for structure-only mode")
            sparse_edges, sparse_types = (
                FrozenPretrainedBiViewEncoder.dense_graph_to_sparse(
                    adj,
                    edge_type,
                )
            )
            encoder_batch["edge_index"] = sparse_edges
            encoder_batch["edge_type"] = sparse_types

        no_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        return self.encoder.predict_with_mask(encoder_batch, no_mask).detach()


class FrozenSingleViewSelfAttention(nn.Module):
    """TPS-aware self-attention classifier input for one frozen view."""

    def __init__(self, encoder_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = encoder_dim * 2
        if self.attention_dim % num_heads != 0:
            raise ValueError("2 * encoder_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.input_dim = encoder_dim + 1
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.W_q = nn.Linear(self.input_dim, self.attention_dim)
        self.W_k = nn.Linear(self.input_dim, self.attention_dim)
        self.W_v = nn.Linear(self.input_dim, self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, view_tokens, TPS, mask=None):
        if TPS.shape != view_tokens.shape[:2]:
            raise ValueError(
                f"Expected TPS [B,L]={tuple(view_tokens.shape[:2])}, "
                f"got {tuple(TPS.shape)}"
            )

        batch_size, seq_len, _ = view_tokens.shape
        features = torch.cat(
            [view_tokens, TPS.unsqueeze(-1).to(view_tokens.dtype)],
            dim=-1,
        )
        features = self.layer_norm(features)

        Q = self.W_q(features).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        K = self.W_k(features).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        V = self.W_v(features).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(
                ~key_mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.attention_dim,
        )
        output = self.out_proj(output)

        if mask is not None:
            output = output.masked_fill(
                ~mask.bool().unsqueeze(-1),
                torch.finfo(output.dtype).min,
            )
        sentence_vector, _ = output.max(dim=1)

        token_norm = view_tokens.float().norm(dim=-1) / (self.encoder_dim ** 0.5)
        if mask is not None:
            metric_mask = mask.to(token_norm.dtype)
            view_stat = (
                (token_norm * metric_mask).sum(dim=1)
                / metric_mask.sum(dim=1).clamp_min(1.0)
            )
        else:
            view_stat = token_norm.mean(dim=1)
        return sentence_vector, attn_weights, view_stat.detach()


class FrozenPretrainedSingleViewDetector(nn.Module):
    def __init__(
        self,
        view_type,
        seq_ckpt=None,
        graph_ckpt=None,
        roberta_name="models/roberta-base",
        num_relations=45,
        num_class=2,
        num_heads=4,
        dropout=0.6,
    ):
        super().__init__()
        self.view_type = view_type
        self.frozen_encoder = FrozenPretrainedSingleViewEncoder(
            view_type=view_type,
            seq_ckpt=seq_ckpt,
            graph_ckpt=graph_ckpt,
            roberta_name=roberta_name,
            num_relations=num_relations,
        )
        self.attention = FrozenSingleViewSelfAttention(
            encoder_dim=self.frozen_encoder.output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.frozen_encoder.output_dim, num_class)

    def train(self, mode=True):
        super().train(mode)
        self.frozen_encoder.eval()
        return self

    def forward(self, input_ids, TPS, adj, edge_type=None, mask=None):
        if mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        view_tokens = self.frozen_encoder(
            input_ids=input_ids,
            attention_mask=mask,
            adj=adj,
            edge_type=edge_type,
        )
        sentence_vector, attn_weights, view_stat = self.attention(
            view_tokens=view_tokens,
            TPS=TPS,
            mask=mask,
        )
        logits = self.fc(sentence_vector)
        zero_metric = view_stat.new_zeros(())
        return logits, attn_weights, view_stat, zero_metric

    def downstream_state_dict(self):
        return {
            "attention": self.attention.state_dict(),
            "fc": self.fc.state_dict(),
        }

    def load_downstream_state_dict(self, state_dict):
        self.attention.load_state_dict(state_dict["attention"], strict=True)
        self.fc.load_state_dict(state_dict["fc"], strict=True)
