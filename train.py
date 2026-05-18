import torch
from utils.data_loader import TextDataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from models.models import BiLSTM_RGCN
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, auc, f1_score, recall_score, precision_score, roc_auc_score
import json


def evaluate(model, bert_model, dataloader, device):
    """
    Evaluate model performance on validation or test set.
    
    Args:
        model: BiLSTM_RGCN model instance
        bert_model: Pretrained BERT model for embeddings
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_dec_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, ncols = 100):
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
            loss = loss_cls+ 1.0*decouple_loss

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_dec_loss += decouple_loss.item()

            probs = torch.softmax(logits, dim = -1)[:, -1]
            preds = torch.argmax(logits, dim = -1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_dec_loss = total_dec_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels , all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    auroc = roc_auc_score(all_labels, all_probs)

    return{
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'dec_loss': avg_dec_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }



def train(args):
    """
    Main training function for BiLSTM-RGCN model.
    
    Handles:
    - Dataset loading and preprocessing
    - Model initialization and training
    - Validation and checkpointing
    - Test set evaluation
    
    Args:
        args: Command line arguments containing training configuration
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load datasets and create data loaders
    train_dataset = TextDataset(args.train_path)

    # dep2idx = train_dataset.dep2idx
    # pos2idx = train_dataset.pos2idx
    with open('./checkpoints/dep2idx_stanza.json', 'r', encoding='utf-8') as f:
            dep2idx = json.load(f)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TextDataset(args.val_path)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False)

    test_dataset = TextDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

    # Initialize BERT model for embeddings (frozen during training)
    bert_model = AutoModel.from_pretrained(args.tokenizer).to(device) # type: ignore
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    # Initialize BiLSTM-RGCN model
    model = BiLSTM_RGCN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        rgcn_hidden_dim=args.rgcn_hidden_dim,
        num_class=args.num_class,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        dep_list=list(dep2idx.keys()),
        max_seq_len=args.max_len
    ).to(device)

    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, params={param.numel()}")

    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    best_auc = 0.0
    log_path = os.path.join(args.save_dir, "training_log.jsonl")
    
    print('Training started...')
    for epoch in range(1,args.epochs +1):
        model.train()
        total_loss = 0.0
        num_samples = 0

        # Training loop with progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",ncols=100)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj = batch['adj'].to(device)
            edge_type = batch['edge_type'].to(device)
            labels = batch['label'].to(device)
            TPS = batch['TPS'].to(device)

            batch_size = input_ids.size(0)
            num_samples += batch_size

            # Get BERT embeddings (no gradient computation)
            with torch.no_grad():
                embeddings = bert_model(input_ids,attention_mask = attention_mask).last_hidden_state
            
            # Training step
            optimizer.zero_grad()
            # print('============================================')
            # print(embeddings.dtype)
            logits,decouple_loss = model(embeddings, TPS, adj, edge_type=edge_type, mask=attention_mask)
            decouple_loss = decouple_loss.mean()         
            loss_cls = criterion(logits,labels)
            loss = loss_cls + args.lambda_decouple * decouple_loss
            loss.backward()
            optimizer.step()

            total_loss +=loss.item() * batch_size
            loop.set_postfix(loss = loss.item(),cls=loss_cls.item(),dec=decouple_loss.item())

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / num_samples
        print(f'[Epoch {epoch}] Average Loss: {avg_train_loss:.6f}')
        
        # Evaluate on validation set
        print('Evaluate started...')
        val_metrics = evaluate(model, bert_model, val_loader, device)
        print(f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f}")

        # Save checkpoint for current epoch
        epoch_model_path = os.path.join(args.save_dir,f'bilstm_gcn_epoch{epoch}.pt')
        if isinstance(model,nn.DataParallel):
            torch.save(model.module.state_dict(),epoch_model_path)
        else:
            torch.save(model.state_dict(), epoch_model_path)

        # Log training progress
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            **val_metrics
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry,ensure_ascii=False)+'\n')

        # Update best model if current performance improves
        if val_metrics['auroc'] > best_auc:
            best_auc = val_metrics['auroc']
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"[Epoch {epoch}] Best model updated! (AUROC={best_auc:.4f})")
    
    
    print('Training finished.')
    print(f"Best AUROC: {best_auc:.4f}")

    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    test_best_model(args, bert_model, device, test_loader)


def test_best_model(args, bert_model, device, test_loader):
    """
    Evaluate the best saved model on test set.
    
    Args:
        args: Command line arguments
        bert_model: Pretrained BERT model for embeddings
        device: Device to run evaluation on
        test_loader: DataLoader for test data
    """
    with open('./checkpoints/dep2idx.json', 'r', encoding='utf-8') as f:
            dep_list = json.load(f)
    # dep_list = list(test_loader.dataset.dep2idx.keys())
    model = BiLSTM_RGCN(
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
    state_dict = torch.load(best_model_path, map_location=device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing!")
        model = nn.DataParallel(model)
        # 判断是否需要去掉 module. 前缀
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
    print("Test set evaluation:")
    print(f"Loss: {metrics['loss']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"AUROC: {metrics['auroc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BiLSTM+GCN for LLM-Generated Text Detection')

    parser.add_argument('--train_path', type = str, default = 'datasets/train.pt', help = 'Path to training dataset')
    parser.add_argument('--val_path', type = str, default = 'datasets/val.pt', help = 'Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='datasets/AcademicResearch/test.pt', help='Path to test dataset')
    parser.add_argument('--tokenizer', type = str, default = './models/roberta-base',help = 'BERT tokenizer name or path')
    parser.add_argument('--max_len', type = int, default = 256, help = 'Maximum sequence length')


    parser.add_argument('--input_dim', type = int, default = 768, help = 'BiLSTM input dimension')
    parser.add_argument('--hidden_dim', type = int, default = 768, help = 'BiLSTM hidden dimension')
    parser.add_argument('--rgcn_hidden_dim', type = int, default = 512, help = 'RGCN hidden dimension')
    parser.add_argument('--num_class', type = int, default = 2, help = 'Number of output classes (Human vs LLM-generated)')
    parser.add_argument('--lstm_layers', type = int, default = 2, help = 'Number of BiLSTM layers')
    parser.add_argument('--dropout', type = float, default = 0.6, help = 'Dropout probability')
    parser.add_argument('--lambda_decouple', type = float, default = 1.0, help = '')
    


    parser.add_argument('--batch_size', type = int, default = 128, help = 'Batch size')
    parser.add_argument('--epochs', type = int, default = 30, help = 'Number of training epochs')
    parser.add_argument('--lr' , type = float, default = 1e-5, help = 'Learning rate')
    parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'L2 regularization weight decay')
    parser.add_argument('--save_dir', type = str ,default = './checkpoints/+D', help = 'Model checkpoint directory')

    args = parser.parse_args()
    train(args)
