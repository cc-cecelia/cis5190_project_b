import argparse
import torch
import torch.nn as nn
from torch.cuda import device
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import json
import os
from sklearn.model_selection import train_test_split
from preprocess import TextDataset,prepare_data
from transformers import DistilBertTokenizerFast
from torch.utils.tensorboard import SummaryWriter

from model import Model
from datetime import datetime
from util.arg_parser import train_parse_args
# -------------------
#   log information
# -------------------
# 定义日志保存目录
LOG_DIR = 'models/logs/'
# 创建一个带时间戳的子目录，保证每次训练的日志不互相覆盖
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(LOG_DIR, run_name)
writer = SummaryWriter(log_path)
# TensorBoard 会将日志文件写入到这个目录

# ------------------
# same tokenizer here
# -------------------
_TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def train():
    args = train_parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data prepare
    texts, labels = prepare_data("../data/processed/processed_data.csv")
    texts = list(texts)
    labels = list(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_dataset = TextDataset(X_train, y_train, _TOKENIZER, max_len= 128)
    val_dataset = TextDataset(X_val, y_val, _TOKENIZER, max_len= 128)
    test_dataset = TextDataset(X_test, y_test, _TOKENIZER, max_len= 128)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)

    # 2. Model Initialization
    model = Model(
        use_dapt=args.use_dapt,
        freeze_encoder=args.freeze_encoder,
        dropout_prob=args.dropout,
    )
    model.to(device)


    # 3. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [],
               "train_acc": [],
               "val_acc": []}

    # 3. Training loop
    print("Start training...")

    for epoch in range(args.epochs):
        # train
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            logits = model(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device)
                logits = model(batch)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 4. saving
    save_path = os.path.join("models/weights", args.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    # 保存训练记录 (JSON 文件) 用于画图
    log_name = args.save_name.replace(".pt", ".json")
    with open(os.path.join("models/logs", log_name), "w") as f:
        json.dump(history, f)