# Ablation Study - Systematic Evaluation

## Objective
Systematically evaluate the contribution of each component to model performance through controlled experiments.

## Experiment Design Overview

| Experiment | Model Architecture | Feature Engineering | Pre-training Strategy | Fine-tuning Strategy | Purpose |
|------------|-------------------|--------------------|-----------------------|---------------------|---------|
| **Baseline** | Logistic Regression | Bag-of-Words | None | None | Establish performance baseline |
| **Exp 1** | DistilBERT | BERT Tokenization | Pre-trained weights | Classifier only (frozen encoder) | Test pre-trained features quality |
| **Exp 2** | DistilBERT + DAPT | BERT Tokenization | Pre-trained + Domain Adaptation | Classifier only (frozen encoder) | Test if domain adaptation improves frozen features |
| **Exp 3** | DistilBERT | BERT Tokenization | Pre-trained weights | End-to-End | Evaluate full model fine-tuning |
| **Exp 4** | DistilBERT + DAPT | BERT Tokenization | Pre-trained + Domain Adaptation | End-to-End | Assess combined domain adaptation + fine-tuning |

---

## Detailed Parameter Configurations

### Baseline: Logistic Regression

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `feature_extraction` | Bag-of-Words (CountVectorizer) | - | Fixed |
| `max_features` | 5000 | [3000, 5000, 10000] | Vocabulary size |
| `ngram_range` | (1, 2) | [(1,1), (1,2), (1,3)] | N-gram context |
| `classifier` | Logistic Regression | - | Fixed |
| `C` (regularization) | 1.0 | [0.1, 1.0, 10.0] | Regularization strength |
| `max_iter` | 1000 | - | Fixed |
| `solver` | lbfgs | - | Fixed |

**Command**: 
```bash
python src/baseline/train_baseline.py --max_features 5000 --ngram_range 1 2 --C 1.0
```

**Purpose**: Establish baseline performance with simple features  
**Expected Result**: Accuracy baseline

---

### Exp 1: DistilBERT (Frozen Encoder + Classifier Fine-tuning)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `model` | distilbert-base-uncased | - | Fixed |
| `freeze_encoder` | True | - | Fixed (frozen) |
| `dropout` | 0.1 | [0.1, 0.3, 0.5] | Regularization |
| `learning_rate` | 1e-3 | [5e-4, 1e-3, 2e-3] | Classifier learning rate |
| `batch_size` | 16 | [8, 16, 32] | Training batch size |
| `epochs` | 5 | [3, 5, 7] | Training duration |
| `optimizer` | AdamW | - | Fixed |
| `weight_decay` | 0.01 | - | Fixed |
| `max_length` | 128 | - | Fixed |

**Command**:
```bash
python src/train.py --freeze_encoder --lr 1e-3 --batch_size 16 --epochs 5 --dropout 0.1
```

**Purpose**: Test quality of pre-trained BERT features  
**Expected Improvement**: +5-10% over Baseline

---

### Exp 2: DistilBERT + DAPT (Frozen Encoder)

#### Phase 1: DAPT (Masked Language Modeling)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `model` | distilbert-base-uncased | - | Fixed |
| `task` | MLM | - | Fixed |
| `training_data` | 14,935 headlines (unlabeled) | - | Fixed |
| `mask_probability` | 0.15 | - | Fixed |
| `learning_rate` | 5e-5 | [2e-5, 5e-5, 1e-4] | DAPT learning rate |
| `batch_size` | 32 | [16, 32, 64] | DAPT batch size |
| `epochs` | 3 | [3, 5, 7] | DAPT duration |
| `optimizer` | AdamW | - | Fixed |
| `save_checkpoint` | models/dapt_checkpoints/ | - | Fixed |

**Command**:
```bash
python src/train_dapt.py --lr 5e-5 --batch_size 32 --epochs 3
```

#### Phase 2: Classifier Training (Frozen DAPT Encoder)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `checkpoint` | dapt_lr5e5_ep3 | - | Best from Phase 1 |
| `use_dapt` | True | - | Fixed |
| `freeze_encoder` | True | - | Fixed (frozen) |
| `dropout` | 0.1 | [0.1, 0.3, 0.5] | Regularization |
| `learning_rate` | 1e-3 | [5e-4, 1e-3, 2e-3] | Classifier learning rate |
| `batch_size` | 16 | [8, 16, 32] | Training batch size |
| `epochs` | 5 | [3, 5, 7] | Training duration |
| `optimizer` | AdamW | - | Fixed |
| `weight_decay` | 0.01 | - | Fixed |
| `max_length` | 128 | - | Fixed |

**Command**:
```bash
python src/train.py --use_dapt --checkpoint dapt_lr5e5_ep3 --freeze_encoder --lr 1e-3 --batch_size 16 --epochs 5 --dropout 0.1
```

**Purpose**: Test if domain adaptation improves frozen feature quality  
**Expected Improvement**: +1-3% over Exp 1 (tests DAPT value without fine-tuning)

---

### Exp 3: DistilBERT (End-to-End Fine-tuning)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `model` | distilbert-base-uncased | - | Fixed |
| `freeze_encoder` | False | - | Fixed (trainable) |
| `dropout` | 0.1 | [0.1, 0.3, 0.5] | Regularization |
| `learning_rate` | 2e-5 | [1e-5, 2e-5, 5e-5] | Full model learning rate |
| `batch_size` | 16 | [8, 16, 32] | Training batch size |
| `epochs` | 5 | [3, 5, 7] | Training duration |
| `optimizer` | AdamW | - | Fixed |
| `weight_decay` | 0.01 | - | Fixed |
| `warmup_ratio` | 0.1 | - | Fixed |
| `max_length` | 128 | - | Fixed |

**Command**:
```bash
python src/train.py --lr 2e-5 --batch_size 16 --epochs 5 --dropout 0.1
```

**Purpose**: Evaluate end-to-end fine-tuning effectiveness  
**Expected Improvement**: +3-5% over Exp 1 (tests fine-tuning value)

---

### Exp 4: DistilBERT + DAPT (End-to-End Fine-tuning)

#### Phase 1: DAPT (Masked Language Modeling)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `model` | distilbert-base-uncased | - | Fixed |
| `task` | MLM | - | Fixed |
| `training_data` | 14,935 headlines (unlabeled) | - | Fixed |
| `mask_probability` | 0.15 | - | Fixed |
| `learning_rate` | 5e-5 | [2e-5, 5e-5, 1e-4] | DAPT learning rate |
| `batch_size` | 32 | [16, 32, 64] | DAPT batch size |
| `epochs` | 3 | [3, 5, 7] | DAPT duration |
| `optimizer` | AdamW | - | Fixed |
| `save_checkpoint` | models/dapt_checkpoints/ | - | Fixed |

**Command**:
```bash
python src/train_dapt.py --lr 5e-5 --batch_size 32 --epochs 3
```

#### Phase 2: Fine-tuning (using DAPT weights)

| Parameter | Value | Alternatives Tested | Control Variable |
|-----------|-------|---------------------|------------------|
| `checkpoint` | dapt_lr5e5_ep3 | - | Best from Phase 1 |
| `use_dapt` | True | - | Fixed |
| `freeze_encoder` | False | - | Fixed (trainable) |
| `dropout` | 0.1 | [0.1, 0.3, 0.5] | Regularization |
| `learning_rate` | 2e-5 | [1e-5, 2e-5, 5e-5] | Fine-tuning learning rate |
| `batch_size` | 16 | [8, 16, 32] | Training batch size |
| `epochs` | 5 | [3, 5, 7] | Fine-tuning duration |
| `optimizer` | AdamW | - | Fixed |
| `weight_decay` | 0.01 | - | Fixed |
| `warmup_ratio` | 0.1 | - | Fixed |
| `max_length` | 128 | - | Fixed |

**Command**:
```bash
python src/train.py --use_dapt --checkpoint dapt_lr5e5_ep3 --lr 2e-5 --batch_size 16 --epochs 5 --dropout 0.1
```

**Purpose**: Assess domain adaptation for news text understanding  
**Expected Improvement**: +1-3% over Exp 3 (tests DAPT value for fine-tuning)

---

## Summary

This ablation study systematically evaluates each component's contribution to model performance through controlled experiments. Key aspects:

### Controlled Variables
All experiments use **consistent training settings** for fair comparison:
- **Batch Size**: 16 (alternatives: [8, 16, 32])
- **Epochs**: 5 (alternatives: [3, 5, 7])
- **Dropout**: 0.1 (alternatives: [0.1, 0.3, 0.5])
- **Max Length**: 128 tokens
- **Optimizer**: AdamW with weight_decay=0.01
- **Warmup Ratio**: 0.1 (for fine-tuning experiments)

**Variable Learning Rates** (adjusted per experiment type):
- **Frozen Encoder (Exp 1, 2)**: 1e-3 (alternatives: [5e-4, 1e-3, 2e-3])
- **Fine-tuning (Exp 3, 4)**: 2e-5 (alternatives: [1e-5, 2e-5, 5e-5])
- **DAPT (Exp 2, 4)**: 5e-5 (alternatives: [2e-5, 5e-5, 1e-4])

### Expected Progressive Improvements
1. **Baseline → Exp 1**: +5-10% (Pre-trained features)
2. **Exp 1 → Exp 2**: +1-3% (Domain adaptation on frozen encoder)
3. **Exp 1 → Exp 3**: +3-5% (End-to-end fine-tuning)
4. **Exp 3 → Exp 4**: +1-3% (Domain adaptation + fine-tuning)

### Key Comparisons
- **Exp 1 vs Exp 2**: Isolates DAPT contribution without fine-tuning
- **Exp 1 vs Exp 3**: Isolates fine-tuning contribution without DAPT
- **Exp 2 vs Exp 4**: Tests if fine-tuning adds value after DAPT
- **Exp 3 vs Exp 4**: Tests if DAPT adds value to fine-tuning

### Evaluation Strategy
- **Data Split**: 80% train / 10% val / 10% test (stratified, seed=42)
- **Metrics**: Accuracy, F1-Score, Precision/Recall (per class), Training Time, Inference Speed
- **Reproducibility**: Same data split, random seed, and evaluation protocol across all experiments

---

## Evaluation Metrics

All experiments use identical metrics and data splits for fair comparison:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Accuracy** | Overall classification accuracy | Correct predictions / Total predictions |
| **Precision (Fox)** | Precision for Fox News class | TP_Fox / (TP_Fox + FP_Fox) |
| **Precision (NBC)** | Precision for NBC News class | TP_NBC / (TP_NBC + FP_NBC) |
| **Recall (Fox)** | Recall for Fox News class | TP_Fox / (TP_Fox + FN_Fox) |
| **Recall (NBC)** | Recall for NBC News class | TP_NBC / (TP_NBC + FN_NBC) |
| **F1-Score (Macro)** | Macro-averaged F1 score | 2 × (Precision × Recall) / (Precision + Recall) |
| **Training Time** | Total training duration | Minutes |
| **Inference Speed** | Prediction throughput | Samples per second |
| **Model Size** | Model parameter count | Millions of parameters |

### Data Split Configuration
- **Training Set**: 80% (~11,948 samples)
- **Validation Set**: 10% (~1,494 samples)  
- **Test Set**: 10% (~1,493 samples)
- **Random Seed**: 42 (for reproducibility)
- **Stratified Split**: Yes (maintain class balance)
