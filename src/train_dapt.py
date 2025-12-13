import argparse
import os
import torch
from torchgen.static_runtime.generator import generate_test_value_names
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from transformers import LineByLineTextDataset
import pandas as pd
from util.arg_parser import datp_parse_args
import config

args = datp_parse_args()

def generate_name():
    name = "dapt"
    if args.lr != config.LR:
        name += "_" + args.lr
    if args.epochs != config.EPOCHS:
        name += "_" + args.epochs
    return name

OUTPUT_NAME = generate_name()
OUTPUT_DIR = config.MODELS_DAPT_CPS / OUTPUT_NAME

def train_dapt():
    # 1. 纯文本数据
    df = pd.read_csv(config.PROCESSED_DATA)
    text_file = config.PROCESSED_DATA_DIR / "dapt_titles.txt"

    with open(text_file, "w", encoding="utf-8") as f:
        for text in df['title'].dropna():
            f.write(str(text) + "\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.BERT)
    model = DistilBertForMaskedLM.from_pretrained(config.BERT)

    dataset = LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path=text_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,  # 3-5 个 epoch
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,  # 我们只需要降低 Loss，不需要准确率
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # 如果有 GPU 开启混合精度加速
        logging_dir=f'../models/logs/{OUTPUT_NAME}_logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting DAPT training...")
    trainer.train()

    # 保存最终模型骨架
    print(f"Saving DAPT model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # if os.path.exists(text_file):
    #     os.remove(text_file)

if __name__ == "__main__":
    train_dapt()