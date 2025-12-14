import torch
from torch import nn
from typing import Any, Iterable, List
from transformers import DistilBertModel, DistilBertTokenizerFast
import os

import sys
sys.path.append("..")
import config

_TOKENIZER = DistilBertTokenizerFast.from_pretrained(config.BERT)
backbone_base = config.MODELS_DIR

class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, use_dapt: bool = False, freeze_encoder: bool = False, dropout_prob: float = 0.1, checkpoint = None, **kwargs) -> None:
        # Initialize your model here
        super().__init__()

        if use_dapt:
            path = config.MODELS_DAPT_CPS / checkpoint
            if os.path.exists(path):
                load_path = path # 本地 用指定的dapt backbone
            else:
                load_path = config.BERT # 非本地，fallback到 plain bert backbone
        else:
            path = config.MODELS_BASE_CPS
            if os.path.exists(path):
                load_path = path # 本地 不用dapt 免下载
            else:
                load_path = config.BERT # 管你本不本地，直接plain bert backbone

        print(f"Initializing model backbone from: {load_path}")

        self.bert = DistilBertModel.from_pretrained(load_path)

        hidden_size = self.bert.config.hidden_size  # 768 for distilbert-base

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2) # 分类头
        self.frozen = freeze_encoder
        # if freeze_encoder: # 仅训练头
        #     for p in self.bert.parameters():
        #         p.requires_grad = False

    def forward(self, batch) -> List[Any]:
        device = next(self.parameters()).device
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        else:
            raise ValueError(f"Wrong batch type: {type(batch)}")

        if input_ids is None:
            raise ValueError("dict input to model.forward missing 'input_ids'")
        if attention_mask is None:
            # construct attention mask if missing
            attention_mask = (input_ids != _TOKENIZER.pad_token_id).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 相当于 pooled output: DistilBertModel 没有 pooler, 通常选择 first token
        hidden = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # [B, 2]
        return logits

    def eval(self) -> None:
        # Optional: set your model to evaluation mode
        self.bert.eval()
        self.classifier.eval()
        return None

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        texts = [example['text'] for example in batch]

        inputs = _TOKENIZER(
            texts,  # List of strings
            padding=True,
            truncation=True,
            max_length=128,  # 和 Dataset 保持一致的上限，maybe可以改成64？
            return_tensors="pt"
        )

        device = next(self.bert.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = self.forward(inputs)
            logits = torch.tensor(logits)
            preds = torch.argmax(logits, dim=-1)
            return preds.tolist()


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


