import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from models.models import SSD
from utils.data_loader import TextDataset
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import argparse
import os
import json
import csv


def evaluate(model, bert_model, dataloader, device, save_alpha_path=None):
    '''
    Evaluation function: computes loss, accuracy, f1, recall, precision, and AUROC metrics.
    
    Args:
        model: Trained BiLSTM_RGCN model
        bert_model: Pretrained BERT model for embeddings
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation metrics
    '''
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_dec_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj = batch['adj'].to(device)
            edge_type = batch['edge_type'].to(device)
            labels = batch['label'].to(device)
            TPS = batch['TPS'].to(device)

            embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
            logits, decouple_loss= model(embeddings, TPS, adj, edge_type=edge_type, mask=attention_mask)
            decouple_loss = decouple_loss.mean()
            loss_cls = criterion(logits, labels)
            loss = loss_cls + 0.1 * decouple_loss
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_dec_loss += decouple_loss.item()

            probs = torch.softmax(logits, dim=-1)[:, -1]
            preds = torch.argmax(logits, dim=-1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())




    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_dec_loss = total_dec_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    auroc = roc_auc_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'dec_loss': avg_dec_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def load_and_evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    with open('./checkpoints/dep2idx_stanza.json', 'r', encoding='utf-8') as f:
            dep_list = json.load(f)
    # with open('./checkpoints/pos2idx.json','r',encoding='utf-8') as f:
    #         pos_list = json.load(f)
    # train_dataset = TextDataset(args.train_path, tokenizer_name=args.tokenizer, max_len=args.max_len)
    test_dataset = TextDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



    bert_model = AutoModel.from_pretrained(args.tokenizer).to(device)
    bert_model.eval()
    for p in bert_model.parameters():
        p.requires_grad = False

    # dep_list = list(train_dataset.dep2idx.keys())
    dep_list = list(dep_list.keys())
    model = SSD(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        rgcn_hidden_dim=args.rgcn_hidden_dim,
        num_class=args.num_class,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        dep_list=dep_list,
        max_seq_len=args.max_len
    ).to(device)
    
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    assert os.path.exists(best_model_path), f"Model weight file not found: {best_model_path}"

    state_dict = torch.load(best_model_path, map_location=device)

    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
        model = nn.DataParallel(model)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    
    metrics = evaluate(model, bert_model, test_loader, device)
    print(f"Evaluation Results on Test Set {args.test_path}:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BiLSTM+GCN on Test Dataset')

    parser.add_argument('--test_path', type=str, default='datasets/L2R/L2R_llm.pt', help='Path to test dataset')
    parser.add_argument('--train_path', type = str, default = './datasets/train.pt', help = 'Path to training dataset (for vocabulary reference)')
    parser.add_argument('--tokenizer', type=str, default='./models/roberta-base', help='BERT tokenizer path')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')

    parser.add_argument('--input_dim', type=int, default=768, help='BiLSTM input dimension')
    parser.add_argument('--hidden_dim', type=int, default=768, help='BiLSTM hidden dimension')
    parser.add_argument('--rgcn_hidden_dim', type=int, default=512, help='RGCN hidden dimension')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classification classes')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')

    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints/+D', help='Model checkpoint directory')

    args = parser.parse_args()
    load_and_evaluate(args)