import argparse

def train_parse_args():
    parser = argparse.ArgumentParser(description="Train News Classifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型backbone Checkpoint。e.g.dapt_lr2e5_ep3")
    parser.add_argument("--use_dapt", action="store_true", help="是否使用 DAPT 预训练过的权重")
    parser.add_argument("--freeze_encoder", action="store_true", help="是否冻结 BERT，只训练分类头")

    parser.add_argument("--warm_lr", type=float, default=2e-5, help="线性层学习率 (Learning Rate) 通用一阶段endtoend 和 frozen")
    parser.add_argument("--bert_lr", type=float, default=2e-5, help="end to end 二阶段学习率")

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率 (传给 model.py)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--bert_ep", type=int, default=3, help="bert + 二阶段线性层 训练轮数")
    parser.add_argument("--warm_ep", type=int, default=3, help="线性层训练轮数")
    parser.add_argument("--memo", type=str, default=None, help="模型备注")

    return parser.parse_args()

def test_parse_args():
    parser = argparse.ArgumentParser(description="Evaluate News Classifier")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .pt file (e.g., models/weights/best_model_dapt.pt)")

    parser.add_argument("--use_dapt", action="store_true",
                        help="Add this flag if the model was trained with DAPT weights")

    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Add this flag if the model was trained with frozen encoder")

    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")

    return parser.parse_args()

def datp_parse_args():
    parser = argparse.ArgumentParser(description="Train DATP")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数(3-5)")
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率 (Learning Rate)")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size per device")
    parser.add_argument("--memo", type=str, default=None, help="模型备注")
    return parser.parse_args()