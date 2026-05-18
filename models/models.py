import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import csv

class StructureSemanticDiscrepancyAttention(nn.Module):
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.combined_dim = hidden_dim * 2
        self.head_dim = self.combined_dim // num_heads
        
        self.scale = self.head_dim **-0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )

        self.W_q = nn.Linear(self.combined_dim, self.combined_dim)
        self.W_k = nn.Linear(self.combined_dim, self.combined_dim)
        self.W_v = nn.Linear(self.combined_dim, self.combined_dim)

        self.out_proj = nn.Linear(self.combined_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.combined_dim)

    def forward(self, x_seq, x_struct, mask=None):
        batch_size, seq_len, _ = x_seq.size()

        h_seq = self.seq_proj(x_seq)
        h_struct = self.struct_proj(x_struct)

        cos_sim = F.cosine_similarity(h_seq, h_struct, dim=-1)
        if mask is not None:
            valid_tokens = mask.sum()
            decouple_loss = (torch.abs(cos_sim) * mask).sum() / (valid_tokens + 1e-8)
        else:
            decouple_loss = torch.abs(cos_sim).mean()

        diff_feat = torch.abs(h_seq - h_struct)
        avg_diff_scalar_per_sample = diff_feat.mean(dim=1).mean(dim=1)  


        gate_input = torch.cat([h_struct, diff_feat], dim=-1)
        alpha = self.gate_net(gate_input)

        enhanced_feat =  (1 - alpha) * h_struct + alpha * h_seq

        combined_feat = torch.cat([enhanced_feat, diff_feat], dim=-1)
        combined_feat = self.layer_norm(combined_feat)

        Q = self.W_q(combined_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(combined_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(combined_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))* self.scale

        if mask is not None:
            # mask shape: [batch, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V) 
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            mask_for_pooling = mask.unsqueeze(-1)
            out = out.masked_fill(mask_for_pooling == 0, -1e9)
        
        sent_vec, _ = out.max(dim=1)
        return sent_vec, attn_weights , avg_diff_scalar_per_sample, decouple_loss


class BiLSTM_RGCN(nn.Module):
    '''
    BiLSTM with Relational Graph Convolutional Network for text classification.

    This model combines:
    - BiLSTM for sequential text encoding.
    - RGCN for depedency graph structure processing
    - POS tag embeddings for linguistic features

    Args:
        - input_dim(int): Dimension of input word embeddings
        - hidden_dim(int): Hidden dimension for LSTM
        - rgcn_hidden_dim(int): Hidden dimension for RGCN layers
        - num_class(int): Number of output classes
        - lstm_layers(int): Number of LSTM layers
        - dropout(float): Dropout rate
        - dep_list(list): List of dependency relations for RGCN
        - pos_size(int): Vocabulary size for POS tags
        - max_seq_len(int): Maximum sequence length
        - pos_dim(int): Dimension of POS tag embeddings
    '''
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                rgcn_hidden_dim, 
                num_class,
                lstm_layers=1, 
                dropout=0.5, 
                dep_list=None, 
                max_seq_len=512):
        super(BiLSTM_RGCN, self).__init__()
        self.max_seq_len = max_seq_len

        # Bidirectional LSTM for sequence encoding
        self.bilstm = nn.LSTM(
            input_size=input_dim+1,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(2 * hidden_dim)

        # RGCN layers for dependency graph processing
        num_relations = len(dep_list) if dep_list else 1
        self.rgcn1 = RGCNConv(input_dim+1, rgcn_hidden_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(rgcn_hidden_dim, rgcn_hidden_dim, num_relations=num_relations)

        # Layer normalization for RGCN outputs
        self.rgcn_norm1 = nn.LayerNorm(rgcn_hidden_dim)
        self.rgcn_norm2 = nn.LayerNorm(rgcn_hidden_dim)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        att_internal_dim = rgcn_hidden_dim
        self.attention = StructureSemanticDiscrepancyAttention(
            seq_dim=2 * hidden_dim,       # BiLSTM 
            struct_dim=rgcn_hidden_dim,   # RGCN 
            hidden_dim=att_internal_dim,  # Attention 
            dropout=dropout
        )
        self.fc = nn.Linear(rgcn_hidden_dim, num_class)

        # BiLSTM-only
        # self.fc = nn.Linear(2 * hidden_dim, num_class)

        # Dependency relation mappings and weights
        self.dep2idx = {dep: idx for idx, dep in enumerate(dep_list)} if dep_list else {}
        self.dep_weights = nn.Parameter(torch.ones(len(dep_list))) if dep_list else None

        

    def forward(self, x, TPS, adj, edge_type=None, mask=None):
        '''
        Forward pass of the BiLSTM-RGCN model.

        Args:
            - x(Tensor): Input word embeddings [batch_size, seq_len, input_dim]
            - pos_idx(Tensor): POS tag indices [batch_size, seq_len]
            - adj(Tensor): Adjacency matrices [batch_size, seq_len, seq_len]
            - edge_type(Tensor): Dependency type indices [batch_size,seq_len, seq_len]
            - mask(Tensor): Attention mask for valid tokens [batch_size, seq_len]
        
        Returns:
            - Tensor: Output logits [batch_size, num_class]
        '''
        
        tps_feature = TPS.unsqueeze(-1).to(x.dtype)
        x_combined = torch.cat([x, tps_feature], dim=-1)

        batch_size, seq_len, _ = x_combined.size()

        # Process sequence with BiLSTM
        h_seq, _ = self.bilstm(x_combined)
        h_seq = self.lstm_norm(h_seq)

        # # BiLSTM-only
        # h_seq = F.max_pool1d(h_seq.permute(0, 2, 1), kernel_size=seq_len).squeeze(-1)
        # h_seq = self.dropout(h_seq)


        rgcn_outputs = []

        # Process each sample in batch separately with RGCN
        for i in range(batch_size):
            node_feat = x_combined[i]
            adj_matrix = adj[i]

            # Extract edge indices from adjacency matrix
            src, dst = adj_matrix.nonzero(as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)

            # Get edge types (dependency relations)
            if edge_type is not None:
                edge_type_i = edge_type[i, src, dst]
            else:
                edge_type_i = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            
            # First RGCN layer
            x_rgcn = self.rgcn1(node_feat, edge_index, edge_type_i)
            x_rgcn = self.rgcn_norm1(x_rgcn)
            x_rgcn = F.elu(x_rgcn)
            x_rgcn = self.dropout(x_rgcn)

            # Second RGCN layer
            x_rgcn = self.rgcn2(x_rgcn, edge_index, edge_type_i)
            x_rgcn = self.rgcn_norm2(x_rgcn)
            x_rgcn = F.elu(x_rgcn)
            x_rgcn = self.dropout(x_rgcn)

            # # rgcn-only
            # x_rgcn, _ = torch.max(x_rgcn, dim=0) 

            rgcn_outputs.append(x_rgcn)
        

        
        x_struct = torch.stack(rgcn_outputs, dim=0)
        # sent_vec, _, decouple_loss= self.attention(x_seq=h_seq, x_struct=x_struct, mask=mask)

        sent_vec, _, alpha,decouple_loss = self.attention(x_seq=h_seq, x_struct=x_struct, mask=mask)
        with open('ID_llmD.csv', "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i in range(batch_size):
                    alpha_i = alpha[i].cpu().numpy().reshape(-1).tolist()
                    writer.writerow(alpha_i)

        # with open('llmD-T-sne.csv', "a", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     for i in range(batch_size):
        #         # 将 [seq_len, hidden_dim] 
        #         diff_feat_i = alpha[i].cpu().numpy().reshape(-1).tolist()
        #         writer.writerow(diff_feat_i)

        # with open('h1.csv', "a", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     for i in range(batch_size):  
        #         for j in range(alpha.shape[1]):  
        #     
        #             token_diff = alpha[i][j].cpu().numpy().tolist()
        #             writer.writerow(token_diff)

        logits = self.fc(sent_vec)

        # logits = self.fc(h_seq)
        # logits = self.fc(x_struct)
        return logits, decouple_loss

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    input_dim = 768
    hidden_dim = 128
    rgcn_hidden_dim = 64
    n_classes = 2

    x = torch.randn(batch_size, seq_len, input_dim)
    adj = torch.randint(0,2,(batch_size, seq_len, seq_len))
    edge_type = torch.randint(0,3,(batch_size, seq_len, seq_len))

    model = BiLSTM_RGCN(input_dim, hidden_dim, rgcn_hidden_dim, n_classes, dep_list=['dep1','dep2','dep3'])
    logits = model(x, adj, edge_type=edge_type)
    print("logits shape:", logits.shape)


class CrossViewCrossAttention(nn.Module):
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        # 传统交叉注意力：Q来自一个模态，K,V来自另一个模态
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Query
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # Key
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # Value

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, x_struct, mask=None):
        """
        x_seq: [B, L_seq, seq_dim] -
        x_struct: [B, L_struct, struct_dim] 
        mask: [B, L_seq]  (1 for valid tokens)
        """
        batch_size, seq_len, _ = x_seq.size()
        _, struct_len, _ = x_struct.size()

        # 分别投影两个模态
        h_seq = self.seq_proj(x_seq)           # [B, L_seq, H]
        h_struct = self.struct_proj(x_struct)  # [B, L_struct, H]

        cos_sim = F.cosine_similarity(h_seq, h_struct, dim=-1)
        if mask is not None:
            valid_tokens = mask.sum()
            decouple_loss = (torch.abs(cos_sim) * mask).sum() / (valid_tokens + 1e-8)
        else:
            decouple_loss = torch.abs(cos_sim).mean()


        Q = self.W_q(h_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h_struct).view(batch_size, struct_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h_struct).view(batch_size, struct_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, heads, L_seq, L_struct]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: [B, L_seq] -> [B, 1, L_seq, 1] 
            mask_expanded = mask.unsqueeze(1).unsqueeze(3)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-1e9'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, heads, L_seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            mask_for_pooling = mask.unsqueeze(-1)  # [B, L_seq, 1]
            out = out.masked_fill(mask_for_pooling == 0, float('-1e9'))

        sent_vec, _ = out.max(dim=1)  # max-pooling across tokens
        return sent_vec, attn_weights, decouple_loss



class StructureSemanticDiscrepancyAttention1(nn.Module):
    # nodiff
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim **-0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, x_struct, mask=None):
        batch_size, seq_len, _ = x_seq.size()

        h_seq = self.seq_proj(x_seq)
        h_struct = self.struct_proj(x_struct)

        cos_sim = F.cosine_similarity(h_seq, h_struct, dim=-1)
        if mask is not None:
            valid_tokens = mask.sum()
            decouple_loss = (torch.abs(cos_sim) * mask).sum() / (valid_tokens + 1e-8)
        else:
            decouple_loss = torch.abs(cos_sim).mean()

        diff_feat = torch.abs(h_seq - h_struct)

        gate_input = torch.cat([h_struct,h_seq], dim=-1)
        alpha = self.gate_net(gate_input)

        enhanced_feat = (1 - alpha) * h_struct + alpha * h_seq
        enhanced_feat = self.layer_norm(enhanced_feat)

        Q = self.W_q(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))* self.scale

        if mask is not None:
            # mask shape: [batch, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V) 
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            mask_for_pooling = mask.unsqueeze(-1)
            out = out.masked_fill(mask_for_pooling == 0, -1e9)
        
        sent_vec, _ = out.max(dim=1)
        return sent_vec, attn_weights, decouple_loss
    
class StructureSemanticDiscrepancyAttention2(nn.Module):
    # concat
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.merge_proj = nn.Linear(hidden_dim * 2, hidden_dim)


        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, x_struct, mask=None):
        batch_size, seq_len, _ = x_seq.size()

        h_seq = self.seq_proj(x_seq)
        h_struct = self.struct_proj(x_struct)

        cos_sim = F.cosine_similarity(h_seq, h_struct, dim=-1)
        if mask is not None:
            valid_tokens = mask.sum()
            decouple_loss = (torch.abs(cos_sim) * mask).sum() / (valid_tokens + 1e-8)
        else:
            decouple_loss = torch.abs(cos_sim).mean()

        h = torch.cat([h_seq, h_struct], dim=-1)

        h = self.merge_proj(h)
        h = self.layer_norm(h)

        Q = self.W_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        sent_vec, _ = out.max(dim=1)
        return sent_vec, attn_weights, decouple_loss


class StructureSemanticDiscrepancyAttention3(nn.Module):
    # strud  seqstruD
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.merge_proj = nn.Linear(2 * hidden_dim, hidden_dim)


        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, x_struct, mask=None):
        batch_size, seq_len, _ = x_seq.size()

        h_seq = self.seq_proj(x_seq)
        h_struct = self.struct_proj(x_struct)

        cos_sim = F.cosine_similarity(h_seq, h_struct, dim=-1)
        if mask is not None:
            valid_tokens = mask.sum()
            decouple_loss = (torch.abs(cos_sim) * mask).sum() / (valid_tokens + 1e-8)
        else:
            decouple_loss = torch.abs(cos_sim).mean()

        diff_feat = torch.abs(h_seq - h_struct)

        h = torch.cat([h_seq, diff_feat], dim=-1)

        h = self.merge_proj(h)
        h = self.layer_norm(h)

        Q = self.W_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        sent_vec, _ = out.max(dim=1)
        return sent_vec, attn_weights,decouple_loss


class StructureSemanticDiscrepancyAttention4(nn.Module):
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=2,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim **-0.5

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        self.merge_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq, x_struct, mask=None):
        batch_size, seq_len, _ = x_seq.size()

        h_seq = self.seq_proj(x_seq)
        h_struct = self.struct_proj(x_struct)

        diff_feat = torch.abs(h_seq - h_struct)

        gate_input = torch.cat([h_struct, diff_feat], dim=-1)
        alpha = self.gate_net(gate_input)

        enhanced_feat =  (1 - alpha) * h_struct + alpha * h_seq
        enhanced_feat = torch.cat([enhanced_feat, diff_feat], dim=-1)
        enhanced_feat = self.merge_proj(enhanced_feat)
        enhanced_feat = self.layer_norm(enhanced_feat)

        Q = self.W_q(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(enhanced_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))* self.scale

        if mask is not None:
            # mask shape: [batch, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V) 
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if mask is not None:
            mask_for_pooling = mask.unsqueeze(-1)
            out = out.masked_fill(mask_for_pooling == 0, -1e9)
        
        sent_vec, _ = out.max(dim=1)
        return sent_vec, attn_weights , alpha