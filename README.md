# SSD: Label-Free Sequential–Structural Discrepancy for LLM-Generated Text Detection

This repository contains the implementation of **SSD**, a dual-view detector that models the discrepancy between sequential context and dependency structure. Large datasets, pretrained language models, experiment outputs, and learned checkpoints are intentionally excluded from the repository.

## Catalogue

- [Introduction](#introduction)
- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Label-Free Discrepancy Extraction](#label-free-discrepancy-extraction)
- [Adversarial Attack](#adversarial-attack)
- [Expected Performance](#expected-performance)
- [Citation](#citation)

## Introduction

SSD represents each token from two complementary views:

- A **Transformer sequence encoder** captures ordered contextual information.
- A **Relational Graph Convolutional Network (RGCN)** captures dependency relations.

The two encoders are first pretrained on an unlabeled corpus. For randomly masked tokens, each branch independently predicts the same frozen RoBERTa target representation. Human/LLM labels, provenance labels, a detector classifier, and cross-view contrastive negatives are not used during this stage.

After pretraining, the view encoders are frozen for the main detector. For token (i), SSD computes

\[
\mathbf{D}_i = |\mathbf{h}^{seq}_i-\mathbf{h}^{str}_i|,
\qquad
d_i = \operatorname{mean}(\mathbf{D}_i).
\]

The fixed discrepancy feature guides token-level view fusion. The fused representation is concatenated with the token proxy score (TPS), processed by multi-head self-attention, max-pooled, and classified as human-written or LLM-generated text.

The main implementation is in `models/models.py`. The repository also retains the single-view detector classes used by the paper's ablations.

## Environment

The experiments were developed with Python 3.10 and CUDA-enabled PyTorch. Install the Python dependencies with:

```bash
conda create -n ssd python=3.10
conda activate ssd

# Select the PyTorch build appropriate for your CUDA installation.
pip install torch torchvision torchaudio
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

The core dependencies are PyTorch, PyTorch Geometric, Transformers, spaCy, scikit-learn, and tqdm. `langchain-openai` is only needed for the optional rewrite attack.

## Data Preparation

### External models

The repository does not contain external model weights. Place the required models at:

| Model | Purpose | Expected path |
|---|---|---|
| RoBERTa-base | Frozen reconstruction target and tokenizer | `models/roberta-base/` |
| Qwen3 | Optional TPS extraction for downstream data | `models/Qwen3/` |

For example:

```bash
git lfs install
git clone https://huggingface.co/FacebookAI/roberta-base models/roberta-base
```

### Input format

Raw data are JSON arrays. Each item contains `text` and `result`:

```json
[
  {"text": "An example document.", "result": 0},
  {"text": "Another example document.", "result": 1}
]
```

For downstream detection, `0` denotes human text and `1` denotes LLM-generated text. Labels in the unlabeled pretraining corpus are ignored and may be set to a dummy value.

### Preprocess unlabeled pretraining data

Prepare each corpus shard without TPS:

```bash
python prepare_vocabe.py \
  --src_json_path datasets/C4/c4_train_300k_part01-of-03.json \
  --save_pt_path datasets/C4/c4_train_300k_part01-of-03.pt \
  --tokenizer_name models/roberta-base \
  --max_len 256 \
  --dep2idx_path checkpoints/dep2idx.json \
  --no-use_tps
```

Repeat this command for all pretraining shards. The provided training scripts expect files matching:

```text
datasets/C4/c4_train_300k_part*-of-03.pt
```

### Preprocess downstream data

When TPS is used, place Qwen3 at `models/Qwen3/` and enable it during preprocessing:

```bash
python prepare_vocabe.py \
  --src_json_path datasets/train.json \
  --save_pt_path datasets/train.pt \
  --tokenizer_name models/roberta-base \
  --max_len 256 \
  --dep2idx_path checkpoints/dep2idx.json \
  --use_tps
```

Apply the same preprocessing settings to validation and test data. Each `.pt` sample contains RoBERTa token IDs, an attention mask, the dependency adjacency matrix, dependency relation IDs, TPS values, and the binary label.

## Training

Training has three stages. Run all commands from the repository root.

### 1. Pretrain the sequence encoder

```bash
GPU_ID=0 bash train_seq.sh
```

This trains a two-layer Transformer sequence encoder with the label-free masked reconstruction objective and saves:

```text
checkpoints/C4/sequential_transformer_c4_300k.pt
```

### 2. Pretrain the structure encoder

```bash
GPU_ID=1 bash train_str.sh
```

This trains a two-layer RGCN structure encoder with the same frozen target space and saves:

```text
checkpoints/C4/graph_rgcn_c4_300k.pt
```

The sequence and structure pretraining jobs are independent and may be run in parallel on different GPUs.

### 3. Train the downstream detector

Place the processed files at `datasets/train.pt` and `datasets/val.pt`, then run:

```bash
GPU_ID=0 bash train_classifier.sh
```

The main configuration freezes both pretrained view encoders and trains only the discrepancy-guided fusion, multi-head self-attention, and classifier. The best validation-AUROC checkpoint is saved to:

```text
checkpoints/frozen_biview_transformer_300k_final/best_auc_model.pt
```

All shell scripts accept additional command-line arguments. For example:

```bash
GPU_ID=0 bash train_classifier.sh --epochs 20 --batch_size 32
```

## Evaluation

Evaluate the saved detector with:

```bash
GPU_ID=0 bash evaluate.sh \
  --data_path datasets/OOD/M4.pt \
  --threshold 0.41
```

The script reports accuracy, macro precision, macro recall, macro F1, AUROC, a confusion matrix, and the mean frozen discrepancy. The decision threshold affects discrete metrics but does not affect AUROC.

To evaluate a checkpoint moved from another machine, the encoder paths may be supplied explicitly:

```bash
python evaluate.py \
  --data_path datasets/test.pt \
  --ckpt_path checkpoints/frozen_biview_transformer_300k_final/best_auc_model.pt \
  --seq_ckpt checkpoints/C4/sequential_transformer_c4_300k.pt \
  --graph_ckpt checkpoints/C4/graph_rgcn_c4_300k.pt \
  --save_pred_path "" \
  --batch_size 128 \
  --threshold 0.41 \
  --amp \
  --mmap_data
```

## Label-Free Discrepancy Extraction

`trainfree_discrepancy.py` extracts one sentence-level discrepancy value per sample without training a downstream detector. The full input is used at inference; no random masking is applied. Special tokens and padding are excluded before averaging token discrepancies.

```bash
python trainfree_discrepancy.py \
  --data_path datasets/test.pt \
  --human_output_csv results/tf_human_D.csv \
  --llm_output_csv results/tf_llm_D.csv \
  --seq_ckpt checkpoints/C4/sequential_transformer_c4_300k.pt \
  --graph_ckpt checkpoints/C4/graph_rgcn_c4_300k.pt
```

Labels are used only after each score has been computed, to route the score into the human or LLM output file. If LLM text is treated as the positive class while human text has the larger discrepancy, use `-D` as the AUROC score and state the direction explicitly.

## Adversarial Attack

`attack.py` contains the rewrite and decoherence perturbation utilities used for robustness evaluation. Do not commit API credentials or generated attack data.

```bash
# Rewrite attack (requires DS_DEEPSEEK_API_KEY)
python attack.py \
  --mode rewrite \
  --input_path datasets/Attack/input.json \
  --output_path datasets/Attack/rewrite.json

# Local adjacent-token decoherence attack
python attack.py \
  --mode decoherence \
  --input_path datasets/Attack/input.json \
  --output_path datasets/Attack/decoherence.json \
  --swap_threshold 20 \
  --seed 42
```

## Expected Performance

The current 300K-pretraining configuration produced the following AUROC values in the reported run:

| Evaluation setting | AUROC (%) |
|---|---:|
| L2R / in-domain | 96.59 |
| M4 / out-of-domain | 78.39 |

Results may vary with preprocessing, corpus sampling, dependency parsing, TPS extraction, random seed, hardware, and software versions. For paper reporting, use repeated runs and report mean and standard deviation rather than treating the values above as guaranteed outputs.

## Repository Structure

```text
SSD/
├── sequential.py               # Transformer sequence pretraining
├── structual.py                # RGCN structure pretraining
├── train.py                    # Downstream detector training
├── evaluate.py                 # Detector evaluation
├── trainfree_discrepancy.py    # Label-free sentence-D extraction
├── prepare_vocabe.py           # Token/graph/TPS preprocessing
├── attack.py                   # Optional robustness attacks
├── train_seq.sh
├── train_str.sh
├── train_classifier.sh
├── evaluate.sh
├── models/models.py
├── utils/
└── checkpoints/dep2idx.json
```

The filename `structual.py` is retained for compatibility with the released checkpoints and scripts.

## Citation

If you use this code, please cite the accompanying paper. The final BibTeX entry will be added after publication.
