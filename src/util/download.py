import os
from transformers import DistilBertModel, DistilBertTokenizerFast

save_path = "../models/base_distilbert"  # 你自己起的名字

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Downloading DistilBert to {save_path}...")

# 2. 下载并保存 Model
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.save_pretrained(save_path)

# 3. 下载并保存 Tokenizer (别忘了这个，它俩是一对)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(save_path)

print("Done! Model and Tokenizer are saved locally.")