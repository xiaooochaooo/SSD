import argparse
import json
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.dep_parse import nlp, sentence_to_dep_matrix
import random


def sentence_to_dep_matrix_without_tps(sentence):
    """Parse with SpaCy directly; the reconstruction experiment does not use TPS."""
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    seq_len = len(tokens)
    adj = torch.zeros(seq_len, seq_len, dtype=torch.float32).numpy()
    dep_types = [["" for _ in range(seq_len)] for _ in range(seq_len)]

    for token in doc:
        head_idx = token.head.i
        child_idx = token.i
        if head_idx != child_idx:
            adj[head_idx, child_idx] = 1
            adj[child_idx, head_idx] = 1
            dep_types[head_idx][child_idx] = token.dep_
            dep_types[child_idx][head_idx] = token.dep_

    for i in range(seq_len):
        adj[i, i] = 1
        dep_types[i][i] = "self"

    return adj, tokens, dep_types, pos_tags, [0.0] * seq_len

# 假设你的句法分析函数在这里引用
# from your_utils import sentence_to_dep_matrix 

def process_and_save(
    src_json_path, 
    save_pt_path, 
    tokenizer_name='models/roberta-base', 
    max_len=256,
    dep2idx_path='checkpoints/dep2idx.json',
    use_tps=False,
):
    print(f"Processing {src_json_path}...")
    
    # 1. 加载原始数据
    with open(src_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 加载或初始化 dep2idx
    # 注意：通常只在处理训练集时构建 dep2idx，验证/测试集应复用
    if os.path.exists(dep2idx_path):
        with open(dep2idx_path, 'r', encoding='utf-8') as f:
            dep2idx = json.load(f)
    else:
        print("Warning: dep2idx not found. You might need to build it first.")
        dep2idx = {} # 或者在这里添加构建 dep2idx 的逻辑

    # 3. 初始化 Tokenizer
    if "roberta" in tokenizer_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    processed_samples = []

    # 4. 循环处理数据 (耗时部分)
    for item in tqdm(data, desc="Converting to Tensors"):
        text = item['text']
        label = int(item['result'])

        
        try:
            if use_tps:
                adj, tokens, dep_types, pos_tags, TPS = sentence_to_dep_matrix(text)
            else:
                adj, tokens, dep_types, pos_tags, TPS = sentence_to_dep_matrix_without_tps(text)
        except Exception as e:
            print(f"Error processing sentence: {text[:20]}... Error: {e}")
            continue 

        n = min(len(tokens), max_len)
        
        # # 扰动
        # n_perturb = max(1, int(0.2 * n))
        # n_pairs = max(1,n_perturb // 2)
        # token_indices = list(range(n))
        # random.shuffle(token_indices)
        
        # # 每次取两个token 作为一对进行扰动
        # for k in range(n_pairs):
        #     idx1, idx2 = token_indices[2*k], token_indices[2*k+1]
        #     perturb_type = 0

        #     if perturb_type == 0:
        #         #交换父节点
        #         parents1 = ((adj[idx1] > 0).nonzero())[0].tolist()
        #         parents2 = ((adj[idx2] > 0).nonzero())[0].tolist()
        #         if parents1 and parents2:
        #             p1, p2 = parents1[0], parents2[0]
        #             adj[idx1, p1], adj[idx2, p2] = 0, 0
        #             adj[idx1, p2], adj[idx2, p1] = 1, 1
        #             edge_type[idx1, p2], edge_type[idx2, p1] = edge_type[idx2, p2], edge_type[idx1, p1]
        #             edge_type[idx1, p1], edge_type[idx2, p2] = 0, 0

        #     else:
        #         # 交换依存关系类型
        #         parents1 = ((adj[idx1] > 0).nonzero())[0].tolist()
        #         parents2 = ((adj[idx2] > 0).nonzero())[0].tolist()
        #         if parents1 and parents2:
        #             p1, p2 = parents1[0], parents2[0]
        #             edge_type[idx1, p1], edge_type[idx2, p2] = edge_type[idx2, p2], edge_type[idx1, p1]
        

        # Tokenize
        enc = tokenizer(
            tokens[:n],
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_len
        )

        # Align the word-level dependency graph to RoBERTa subword positions.
        # Position 0 is normally <s>, so copying adj[:n, :n] directly would
        # shift every dependency edge and would also ignore wordpiece splits.
        word_ids = enc.word_ids(batch_index=0)
        word_to_subwords = {}
        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < n:
                word_to_subwords.setdefault(word_idx, []).append(subword_idx)

        adj_padded = torch.zeros(max_len, max_len)
        edge_type = torch.zeros(max_len, max_len, dtype=torch.long)
        self_rel = dep2idx.get("self", 0)

        # Give every real subword a self-loop and connect continuation
        # wordpieces to the first wordpiece of the same original token.
        for positions in word_to_subwords.values():
            first = positions[0]
            for pos in positions:
                adj_padded[pos, pos] = 1
                edge_type[pos, pos] = self_rel
                if pos != first:
                    adj_padded[first, pos] = 1
                    adj_padded[pos, first] = 1
                    edge_type[first, pos] = self_rel
                    edge_type[pos, first] = self_rel

        # Dependency edges connect the representative (first) subword of
        # each parsed token.  Continuation pieces receive information through
        # the within-token edges above.
        for i in range(n):
            if i not in word_to_subwords:
                continue
            src = word_to_subwords[i][0]
            for j in range(n):
                if adj[i, j] <= 0 or j not in word_to_subwords:
                    continue
                dst = word_to_subwords[j][0]
                dep = dep_types[i][j]
                adj_padded[src, dst] = 1
                edge_type[src, dst] = dep2idx.get(dep, self_rel)

        # Broadcast each original-token probability to its RoBERTa subwords.
        tps_padded = torch.zeros(max_len)
        for word_idx, positions in word_to_subwords.items():
            if word_idx < len(TPS):
                for pos in positions:
                    tps_padded[pos] = float(TPS[word_idx])

        # 封装样本
        sample = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'adj': adj_padded,
            'edge_type': edge_type,
            'label': torch.tensor(label, dtype=torch.long),
            'TPS': tps_padded
        }
        processed_samples.append(sample)

    # 5. 保存到磁盘
    os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)
    torch.save(processed_samples, save_pt_path)
    print(f"Done! Saved {len(processed_samples)} samples to {save_pt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_json_path", required=True)
    parser.add_argument("--save_pt_path", required=True)
    parser.add_argument("--tokenizer_name", default="models/roberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--dep2idx_path", default="checkpoints/dep2idx.json")
    parser.add_argument(
        "--use_tps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the original Qwen TPS tokenizer path. Keep disabled for label-free reconstruction.",
    )
    args = parser.parse_args()
    process_and_save(
        src_json_path=args.src_json_path,
        save_pt_path=args.save_pt_path,
        tokenizer_name=args.tokenizer_name,
        max_len=args.max_len,
        dep2idx_path=args.dep2idx_path,
        use_tps=args.use_tps,
    )
