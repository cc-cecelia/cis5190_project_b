import os
import torch
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


def generate_name():
    name = "dapt"
    if args.lr != config.LR:
        name += "_" + args.lr
    if args.epochs != config.EPOCHS:
        name += "_" + args.epochs
    return name

def train_dapt(output_dir):
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
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,  # 3-5 个 epoch
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,  # 我们只需要降低 Loss，不需要准确率
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # 如果有 GPU 开启混合精度加速
        logging_dir=f'../models/logs/{output_dir}_logs',
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
    print(f"Saving DAPT model to {output_dir}...")
    if os.path.exists(output_dir):
        output_dir = output_dir + "_temp"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = datp_parse_args()
    name = generate_name()
    output_dir = config.MODELS_DAPT_CPS / name
    train_dapt(output_dir)