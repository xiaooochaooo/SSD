import spacy
import numpy as np
import json
from tqdm import tqdm
import os
import stanza
from spacy.tokens import Doc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

print('Load spacy model...')
nlp = spacy.load("en_core_web_lg")

# print('Load stanza model...')
# nlp = stanza.Pipeline('en', model_dir='models/stanza', processors='tokenize,mwt,pos,lemma,depparse', download_method=None)

def get_token_probability_sequence(model_id, text):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")  # 替换torch_dtype为dtype


    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs["input_ids"]
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_tensor = torch.tensor([[bos_token_id]], device=model.device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)
    # 3. Logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  #  [batch, sequence_length, vocab_size]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    probs = F.softmax(shift_logits, dim=-1)


    token_probs = torch.gather(probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    

    tps_list = token_probs[0].tolist()
    tokens = [tokenizer.decode([t]) for t in shift_labels[0]]

    return tps_list,tokens


def custom_spacy_parser(tokens):

    

    doc = Doc(nlp.vocab, words=tokens)
    
    doc = nlp.get_pipe("parser")(doc)  
    

    return doc

def sentence_to_dep_matrix(sentence):
    '''
    Convert sentence to dependency relation adjacency matrix with type annotations.

    Constructs undirected graph representation of dependency parse where:
    - Adjacency matrix indicates connection between tokens
    - Dependency type matrix stores grammatical relationship labels
    - POS tags provide additional linguistic information

    Args:
        - sentence (str): Input text to parse
    
    Returns:
        - tuple: Contains four elements:
            - adj(np.ndarray): Binary adjacency matrix of dependency relations
            - tokens(List[str]): List of tokens in the sentence
            - dep_types(List[List[str]]): Matrix of dependency relation labels
            - pos_tags(List[str]): Part-of-speech tags for each token
    '''
    l,t = get_token_probability_sequence('models/Qwen3',sentence)
    doc = custom_spacy_parser(t)
    
    # Extract linguistic features
    tokens=[token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    seq_len=len(tokens)

    # Initialize matrices
    adj=np.zeros((seq_len,seq_len),dtype=np.float32)
    dep_types = [['' for _ in range(seq_len)] for _ in range(seq_len)]


    # Build dependency graph -iterate trough each token
    for token in doc:
        head_idx=token.head.i
        child_idx=token.i
        dep_type = token.dep_

        # Skip self-connections and add bidirectional edges for undirected graph
        if head_idx!=child_idx:
            adj[head_idx,child_idx]=1
            adj[child_idx,head_idx]=1
            dep_types[head_idx][child_idx]=dep_type
            dep_types[child_idx][head_idx]=dep_type
    adj = adj + np.eye(seq_len, dtype=np.float32)
    
    for i in range(seq_len):
        dep_types[i][i] = 'self'
    
    return adj,tokens,dep_types,pos_tags,l

# def sentence_to_dep_matrix(sentence):
#     doc = nlp(sentence)
    
#     sent = doc.sentences[0]
    
#     # Extract linguistic features
#     tokens = [word.text for word in sent.words]
#     pos_tags = [word.pos for word in sent.words]
#     seq_len = len(tokens)

#     # Initialize matrices
#     adj = np.zeros((seq_len, seq_len), dtype=np.float32)
#     dep_types = [['' for _ in range(seq_len)] for _ in range(seq_len)]

#     # Build dependency graph

#     for i, word in enumerate(sent.words):
#         # head 
#         head_idx = word.head - 1 if word.head > 0 else i  # head=0 表示根节点
#         child_idx = i
#         dep_type = word.deprel
        
#         # Skip self-connections and add bidirectional edges for undirected graph
#         if head_idx != child_idx and head_idx >= 0:
#             adj[head_idx, child_idx] = 1
#             adj[child_idx, head_idx] = 1
#             dep_types[head_idx][child_idx] = dep_type
#             dep_types[child_idx][head_idx] = dep_type
    
#     return adj, tokens, dep_types, pos_tags


if __name__ == "__main__":
    for dir in os.listdir('./dataset'):
        with open('./dataset/'+dir, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        print(f'loading {dir}...')
        for data in tqdm(datas):
            sentence=data['text']
            adj,tokens,dep_types=sentence_to_dep_matrix(sentence)
            print(f'adj shape: {adj.shape}, tokens: {tokens}, dep_types: {dep_types}')
            