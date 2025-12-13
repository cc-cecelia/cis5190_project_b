import os

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from model import Model
from util.arg_parser import test_parse_args
from preprocess import prepare_data, TextDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

_TOKENIZER_NAME = "distilbert-base-uncased"

args = test_parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

texts, labels = prepare_data("../data/processed/processed_data.csv")

_, X_temp, _, y_temp = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained(_TOKENIZER_NAME)
test_dataset = TextDataset(X_test, y_test, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = Model(
    use_dapt=args.use_dapt,
    freeze_encoder=args.freeze_encoder
)
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found at {args.model_path}")

state_dict = torch.load(args.model_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        # 搬运 Label
        batch_labels = batch['labels'].to(device)
        logits = model(batch)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print("\n" + "=" * 50)
print(f"Final Test Accuracy: {acc:.4f}")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))