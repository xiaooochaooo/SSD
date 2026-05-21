import json
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.dep_parse import sentence_to_dep_matrix
import random

# from your_utils import sentence_to_dep_matrix 

def process_and_save(
    src_json_path, 
    save_pt_path, 
    tokenizer_name='models/roberta-base', 
    max_len=256,
    dep2idx_path='checkpoints/dep2idx.json'
):
    print(f"Processing {src_json_path}...")
    
    with open(src_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if os.path.exists(dep2idx_path):
        with open(dep2idx_path, 'r', encoding='utf-8') as f:
            dep2idx = json.load(f)
    else:
        print("Warning: dep2idx not found. You might need to build it first.")
        dep2idx = {} 


    if "roberta" in tokenizer_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    processed_samples = []


    for item in tqdm(data, desc="Converting to Tensors"):
        text = item['text']
        label = int(item['result'])

        
        try:
            adj, tokens, dep_types, pos_tags, TPS = sentence_to_dep_matrix(text)
        except Exception as e:
            print(f"Error processing sentence: {text[:20]}... Error: {e}")
            continue 

        n = min(len(tokens), max_len)


        edge_type = torch.zeros(max_len, max_len, dtype=torch.long)
        for i in range(n):
            for j in range(n):
                if adj[i,j] > 0:
                    dep = dep_types[i][j]
                    if dep in dep2idx:
                        edge_type[i,j] = dep2idx[dep]
        

        # n_perturb = max(1, int(0.2 * n))
        # n_pairs = max(1,n_perturb // 2)
        # token_indices = list(range(n))
        # random.shuffle(token_indices)
        
        # for k in range(n_pairs):
        #     idx1, idx2 = token_indices[2*k], token_indices[2*k+1]
        #     perturb_type = 0

        #     if perturb_type == 0:
        #         parents1 = ((adj[idx1] > 0).nonzero())[0].tolist()
        #         parents2 = ((adj[idx2] > 0).nonzero())[0].tolist()
        #         if parents1 and parents2:
        #             p1, p2 = parents1[0], parents2[0]
        #             adj[idx1, p1], adj[idx2, p2] = 0, 0
        #             adj[idx1, p2], adj[idx2, p1] = 1, 1
        #             edge_type[idx1, p2], edge_type[idx2, p1] = edge_type[idx2, p2], edge_type[idx1, p1]
        #             edge_type[idx1, p1], edge_type[idx2, p2] = 0, 0

        #     else:
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

        # Pad Adjacency Matrix
        adj_padded = torch.zeros(max_len, max_len)
        adj_padded[:n, :n] = torch.tensor(adj[:n, :n])

        # Pad TPS
        tps_padded = torch.zeros(max_len)
        tps_padded[:n] = torch.tensor(TPS[:n])

        sample = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'adj': adj_padded,
            'edge_type': edge_type,
            'label': torch.tensor(label, dtype=torch.long),
            'TPS': tps_padded
        }
        processed_samples.append(sample)

    os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)
    torch.save(processed_samples, save_pt_path)
    print(f"Done! Saved {len(processed_samples)} samples to {save_pt_path}")

if __name__ == "__main__":
    RAW_TRAIN_FILE = ''  
    SAVE_TRAIN_PT = '' 
    

    process_and_save(RAW_TRAIN_FILE, SAVE_TRAIN_PT)
    