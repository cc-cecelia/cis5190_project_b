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
from util.generate_name import modify_if_exists
from util.arg_parser import datp_parse_args

import sys
sys.path.append("..")
import config
from util.generate_name import generate_dapt_ckp_name

def train_dapt(output_dir, csv_name = "processed_data.csv"):
    # 1. 纯文本数据
    df = pd.read_csv(config.PROCESSED_DIR / csv_name)
    text_file = config.PROCESSED_DIR / "dapt_titles.txt"

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

    whole_ckpt = output_dir.parent / (output_dir.name + "_full")
    whole_ckpt = modify_if_exists(whole_ckpt)
    training_args = TrainingArguments(
        output_dir=whole_ckpt,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,  # 3-5 个 epoch
        per_device_train_batch_size=args.batch_size,
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
    backbone = output_dir.parent / (output_dir.name + "_backbone")
    print(f"Saving DAPT model to {backbone}...")
    backbone = modify_if_exists(backbone)
    trainer.save_model(backbone)
    tokenizer.save_pretrained(backbone)

if __name__ == "__main__":
    args = datp_parse_args()
    name = generate_dapt_ckp_name(args)
    output_dir = config.MODELS_DAPT_CPS / name

    train_dapt(output_dir, args.csv_name)