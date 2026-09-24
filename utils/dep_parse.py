import spacy
import numpy as np
import json
from tqdm import tqdm
import os
from spacy.tokens import Doc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

print('Load spacy model...')
nlp = spacy.load("en_core_web_lg")

def get_token_probability_sequence(model_id, text):
    # 1. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map={"":0})  # 替换torch_dtype为dtype

    # 2. 对文本进行编码
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs["input_ids"]
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_tensor = torch.tensor([[bos_token_id]], device=model.device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)
    # 3. 获取模型的输出 (Logits)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # 形状: [batch, sequence_length, vocab_size]

    # 4. 计算概率序列
    # 我们需要的是第 i 个 token 对第 i+1 个 token 的预测概率
    # 所以 logits 要向后偏移一位，且去掉最后一个预测（因为没有 ground truth）
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # 将 Logits 转换为概率 (Softmax)
    probs = F.softmax(shift_logits, dim=-1)

    # 提取实际出现的 Token 的概率
    # 使用 gather 获取每个位置上对应 label 的概率值
    token_probs = torch.gather(probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # 转换为列表
    tps_list = token_probs[0].tolist()
    tokens = [tokenizer.decode([t]) for t in shift_labels[0]]

    return tps_list,tokens


def custom_spacy_parser(tokens):
    # 使用自定义的分词器对文本进行处理

    # 创建一个新的 Doc 对象
    doc = Doc(nlp.vocab, words=tokens)

    doc = nlp.get_pipe("parser")(doc)  # 使用依赖解析管道

    # 返回解析后的 Doc
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
#     # Stanza 中每个 word 有 head 和 deprel 属性
#     for i, word in enumerate(sent.words):
#         # head 是从1开始的索引，需要转换为0-based
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
