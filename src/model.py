import torch
from torch import nn
from typing import Any, Iterable, List
from transformers import DistilBertModel, DistilBertTokenizerFast

_TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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

    def __init__(self, bert_path: str = None, freeze_encoder: bool = False, dropout_prob: float = 0.1) -> None:
        # Initialize your model here
        super().__init__()

        if bert_path:
            # 从huggingface下载
            self.bert = DistilBertModel.from_pretrained(bert_path)
        else:
            # 从本地加载
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        hidden_size = self.bert.config.hidden_size  # 768 for distilbert-base

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2) # 分类头

        if freeze_encoder: # 仅训练头
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, batch: Iterable[Any]) -> List[Any]:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
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
        return None

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        inputs = self.tokenizer(
            batch,  # List of strings
            padding=True,
            truncation=True,
            max_length=128,  # 和 Dataset 保持一致的上限，maybe可以改成64？
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = self.forward(inputs)
            preds = torch.argmax(logits, dim=-1)
            return preds.tolist()


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


