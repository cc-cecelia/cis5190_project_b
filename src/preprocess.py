import torch
import pandas as pd
from typing import Tuple
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

LABEL_MAP = {
    "FoxNews": 0,
    "NBC": 1
}

def prepare_data(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess CSV for DistilBERT classification.

    Input:
        csv_path: str - path to CSV file containing "title" and "label" columns

    Output:
        X: dict of torch.Tensor -> {
                "input_ids": LongTensor [N, seq_len],
                "attention_mask": LongTensor [N, seq_len]
           }
        y: LongTensor [N]
    """

    # Load CSV
    df = pd.read_csv(csv_path)

    # Expecting dataset to have at least: "title", "label"
    if "title" not in df.columns:
        raise ValueError("CSV must contain a 'title' column.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Convert labels to numeric
    # Backend defines: FoxNews->0, NBC->1
    # TODO: Can be removed if it has already been mapped by Data cleaning
    y = df["label"].map(LABEL_MAP).astype(int).tolist()

    # Tokenize all titles
    encodings = tokenizer(
        df["title"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    # encodings is a dict: {"input_ids": tensor, "attention_mask": tensor}
    X_input_ids = encodings["input_ids"]
    X_attention_mask = encodings["attention_mask"]

    # Combine into a single dict (model.py expects dict input)
    X = {
        "input_ids": X_input_ids,
        "attention_mask": X_attention_mask,
    }

    # Convert y to LongTensor
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X, y_tensor
