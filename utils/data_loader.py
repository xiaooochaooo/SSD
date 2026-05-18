from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.dep_parse import sentence_to_dep_matrix
import torch
from tqdm import tqdm


class TextDataset(Dataset):
    '''
    Custom text dataset for dependency-aware text classification.

    Processes text samples to generate:
    - BERT token encodings with attention masks
    - Dependency parse adjacency matrices  
    - Edge type mappings for grammatical relations
    - Pos tag indices for linguistic features

    Attributes:
        - data(List[Dict]): Loaded dataset from JSON file
        - max_len(int): Maximum sequence length for padding/truncation
        - tokenizer: Pretrained tokenizer instance
        - dep2idx(Dict[str, int]): Dependency type to index mapping
        - pos2idx(Dict[str, int]): POS tag to index mapping
    '''

    def __init__(self, pt_file_path):
        '''
        Initialize dataset and build vocabulary mappings.
        '''

        print(f"Loading data from {pt_file_path}...")
        try:
            self.data = torch.load(pt_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed file not found at {pt_file_path}. Please run preprocess.py first.")
            
        print(f"Successfully loaded {len(self.data)} samples.")
        
        # Save vocabulary mappings for consistent usage
        # json.dump(self.dep2idx, open('dep2idx.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        # json.dump(self.pos2idx, open('pos2idx.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = TextDataset('./dataset/xsum_gpt4.json')
    print("Dataset size:", len(dataset))
    sample = dataset[0]
    
    print("input_ids shape:", sample['input_ids'].shape)
    print("attention_mask shape:", sample['attention_mask'].shape)
    print("adj shape:", sample['adj'].shape)
    print("edge_type shape:", sample['edge_type'].shape)
    print("label:", sample['label'])