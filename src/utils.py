import argparse

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_csv", type=str, required=True)

    # phases
    p.add_argument("--do_dapt", action="store_true")
    p.add_argument("--dapt_epochs", type=int, default=0)
    p.add_argument("--ft_epochs", type=int, default=3)

    # model tricks
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--use_two_fields", action="store_true", help="TF 拼接模式")

    # optimization
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true")

    # system
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_path", type=str, default="../weights/model.pt")

    return p.parse_args()