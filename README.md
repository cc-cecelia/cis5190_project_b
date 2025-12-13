
### 运行顺序

1.  **DAPT** ：如果您使用 DAPT 权重，需要先运行 `train_dapt.py`。
2.  **模型训练**：运行 `train.py` 来训练分类模型。
3.  **模型评估**：运行 `test.py` 来评估训练好的模型性能。

-----

### 1\. DAPT

DAPT 是对distil bert 在文本分布上进行的unsupervised训练，会更改bert的backbone，因此在**代码内部逻辑**中会用到不同的checkpoint文件夹`base_checkpoints`,`dapt_checkpoints`

**运行命令:**

```bash
python train_dapt.py
```

**说明:**
  * 该脚本会使用 `data/processed/processed_data.csv` 中的标题数据，在 DistilBERT 上进行 Masked Language Modeling 任务的继续训练。
  * 训练后的权重将保存在 `models/dapt_checkpoints` 目录下。

-----

### 2\. 模型训练

使用 `train.py` 脚本训练新闻分类模型。

**运行命令示例:**

```bash
# 基础模型训练 (不使用 DAPT, 训练所有层，默认 lr 2e-5，batch_size 16 epoch 3)
python train.py --save_name model_base.pt

# 使用 DAPT 权重训练 (训练所有层)
python train.py --use_dapt --save_name model_dapt.pt --batch_size 16 --epochs 3

# 使用 DAPT 权重并冻结编码器训练 (只训练分类头)
python train.py --use_dapt --freeze_encoder --save_name model_dapt_lr1e5_ep_10_frozen.pt --epochs 10 --lr 1e-5

# 剩下的可以自行组合，注意model save命名，如果不是默认参数，就要写上具体内容
```

**核心参数说明:**

| 参数 | 默认值 | 描述                                                 |
| :--- | :--- |:---------------------------------------------------|
| `--use_dapt` | `False` | **(Flag)** 使用 DAPT 预训练的权重 (需要先运行 `train_dapt.py`)。 |
| `--freeze_encoder` | `False` | **(Flag)** 冻结 BERT 编码器层，只训练顶部的分类头。                 |
| `--lr` | `2e-5` | 学习率 (Learning Rate)。                               |
| `--dropout` | `0.1` | 分类头中的 Dropout 概率。                                  |
| `--batch_size` | `16` | 训练批次大小 (Batch Size)。                               |
| `--epochs` | `3` | 训练轮数。                                              |
| `--save_name` | **(必须)** | 保存的模型文件名，如 `model_base.pt`。                        |

-----

### 3\. 模型评估

使用 `test.py` 脚本评估已训练模型的性能。

**运行命令示例:**

```bash
# 评估使用 DAPT 训练的模型
python test.py --model_path models/weights/model_dapt_XXXX.pt --use_dapt

# 评估基础模型
python test.py --model_path models/weights/model_base_XXXX.pt
```

**核心参数说明:**

| 参数 | 默认值 | 描述                                                 |
| :--- | :--- |:---------------------------------------------------|
| `--model_path` | **(必须)** | 待评估模型的 `.pt` 文件路径。                                 |
| `--use_dapt` | `False` | **(Flag)** 必须与模型名字里是否有dapt设置保持一致，用于正确加载 DAPT 模型架构。 |
