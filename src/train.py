import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import json
import os
from sklearn.model_selection import train_test_split
from preprocess import TextDataset,prepare_data
from transformers import DistilBertTokenizerFast

from model import Model
from util.arg_parser import train_parse_args
from util.generate_name import generate_model_name, modify_if_exists
import sys
sys.path.append("..")
import config

_TOKENIZER = DistilBertTokenizerFast.from_pretrained(config.BERT)

def train():
    args = train_parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # 1. Data prepare
    print("Preparing data...")
    texts, labels = prepare_data(config.PROCESSED_DATA)
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
    # test_dataset = TextDataset(X_test, y_test, _TOKENIZER, max_len= 128)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=2,pin_memory=True)

    print("Preparing model...")
    # 2. Model Initialization
    model = Model(
        use_dapt=args.use_dapt,
        freeze_encoder=args.freeze_encoder,
        dropout_prob=args.dropout,
        checkpoint=args.checkpoint,
    )
    model.to(device)
    if args.freeze_encoder:
        print("仅训练线性层")
    else:
        print("End-to-End 两阶段训练")

    # 无论是不是frozen 的 bert都得先冻上
    for param in model.bert.parameters():
        param.requires_grad = False
    print("Bert 已冻结")
    print("Start training...")
    if not args.freeze_encoder: # 如果是end to end 线性层训练则不计入log
        # only optimize linear layer
        warm_optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.warm_lr)
        warm_criterion = nn.CrossEntropyLoss()

        for ep in range(args.warm_ep):
            model.train()
            total_loss = 0
            for batch in train_loader:
                labels = batch["labels"].to(device)
                logits = model(batch)
                loss = warm_criterion(logits, labels)

                warm_optimizer.zero_grad()
                loss.backward()
                warm_optimizer.step()
                total_loss += loss.item()
            print(f"[Stage 1] Epoch {ep + 1}/{args.warm_ep} | Warmup Loss: {total_loss / len(train_loader):.4f}")

        print("阶段一完成。解冻 Bert 准备进入阶段二...\n")
        for param in model.parameters():
            param.requires_grad = True

    # 现在要么训练linear层或者俩一起训练
    if args.freeze_encoder: # 只有linear，用大的
        lr = args.warm_lr
        epochs = args.warm_ep
    else: # 都得来，用小的
        lr = args.bert_lr
        epochs = args.bert_ep

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [],
               "train_acc": [],
               "val_acc": []}

    # 3. Training loop
    for epoch in range(epochs):
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
                # logits = torch.tensor(logits)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 4. saving
    os.makedirs(config.MODELS_WEIGHTS_DIR, exist_ok=True)
    save_name = generate_model_name(args)
    print(f"Saving model {save_name}...")
    save_path = config.MODELS_WEIGHTS_DIR / save_name
    save_path = modify_if_exists(save_path)
    torch.save(model.state_dict(), save_path)

    print(f"Model weights saved to {save_path}")

    # 保存训练记录 (JSON 文件) 用于画图
    log_name = save_name.replace(".pt", ".json")
    with open(config.MODELS_LOG_DIR / log_name, "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    train()