import argparse

def train_parse_args():
    parser = argparse.ArgumentParser(description="Train News Classifier")

    parser.add_argument("--use_dapt", action="store_true", help="是否使用 DAPT 预训练过的权重")
    parser.add_argument("--freeze_encoder", action="store_true", help="是否冻结 BERT，只训练分类头")

    parser.add_argument("--lr", type=float, default=2e-5, help="学习率 (Learning Rate)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率 (传给 model.py)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")

    parser.add_argument("--save_name", type=str, required=True, help="保存模型的名字，例如 model_dapt_lr2e5.pt 或 model_base_lr2e5_ep10")
    parser.add_argument("--dapt_path", type=str, default="models/dapt_checkpoints", help="DAPT 权重的本地文件夹路径")

    return parser.parse_args()

def test_parse_args():
    parser = argparse.ArgumentParser(description="Evaluate News Classifier")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .pt file (e.g., models/weights/best_model_dapt.pt)")

    parser.add_argument("--use_dapt", action="store_true",
                        help="Add this flag if the model was trained with DAPT weights")

    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Add this flag if the model was trained with frozen encoder")

    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--dapt_path", type=str, default="models/dapt_checkpoints", help="Path to DAPT folder")

    return parser.parse_args()