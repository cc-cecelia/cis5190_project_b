import sys
sys.path.append("..")
import config

def format_lr_name(lr_value):
    """
    将学习率（例如 0.00001 或 1e-05）转换为简洁的字符串格式（例如 '1e5'）。
    """
    # 这一步使用 f-string 格式化，强制转换为科学计数法，得到 '1e-05'
    lr_str = f"{lr_value:.0e}"

    # 移除 'e-' 或 'e+ ' 中的符号，得到 '1e5'
    if 'e-' in lr_str:
        return lr_str.replace('e-', 'e')
    else:
        return str(lr_str)


# def generate_model_name(args):
#     # model_dapt_lr1e5_ep10_frozen.pt
#     name = "model"
#     if args.use_dapt:
#         name += "_" + args.checkpoint + "_FT"
#     else:
#         name += "_base"
#
#     if args.lr != config.LR:
#         formatted_lr = format_lr_name(args.lr)
#         name += "_lr" + formatted_lr
#
#     if args.batch_size != config.BATCH_SIZE:
#         name += "_bs" + str(args.batch_size)
#
#     if args.dropout != config.DROPOUT:
#         # 1. 转换为字符串并移除小数点 (例如: '0.1' -> '01')
#         dropout_str_raw = str(args.dropout).replace('.', '')
#
#         # 2. 移除前导零 (例如: '01' -> '1')
#         dropout_str_clean = dropout_str_raw.lstrip('0')
#         if not dropout_str_clean:  # args.dropout == 0.0
#             dropout_str_clean = '0'
#         name += "_do" + dropout_str_clean
#
#     if args.epochs != config.EPOCHS:
#         name += "_ep" + str(args.epochs)
#
#     if args.freeze_encoder:
#         name += "_frozen"
#
#     if args.memo is not None:
#         name += args.memo
#
#     name += ".pt"
#     return name

def generate_model_name(args):
    # model_dapt_lr1e5_ep10_frozen.pt
    name = "model"
    if args.use_dapt:
        name += "_" + args.checkpoint + "_FT"
    else:
        name += "_base"

    formatted_warm_lr = format_lr_name(args.warm_lr)
    name += "_warmlr" + formatted_warm_lr

    if not args.freeze_encoder:
        formatted_bert_lr = format_lr_name(args.bert_lr)
        name += "_bertlr" + formatted_bert_lr

    name += "_bs" + str(args.batch_size)

        # 1. 转换为字符串并移除小数点 (例如: '0.1' -> '01')
    dropout_str_raw = str(args.dropout).replace('.', '')

        # 2. 移除前导零 (例如: '01' -> '1')
    dropout_str_clean = dropout_str_raw.lstrip('0')
    if not dropout_str_clean:  # args.dropout == 0.0
        dropout_str_clean = '0'
    name += "_do" + dropout_str_clean


    name += "_warmep" + str(args.warm_ep)

    if not args.freeze_encoder:
        name += "_bertep" + str(args.bert_ep)

    if args.freeze_encoder:
        name += "_frozen"

    if args.memo is not None:
        name += "_"+args.memo

    name += ".pt"
    return name

# def generate_dapt_ckp_name(args):
#     name = "dapt"
#
#     # *** 关键修改：使用 format_lr_name 函数处理学习率 ***
#     if args.lr != 3e-5:
#         formatted_lr = format_lr_name(args.lr)
#         name += "_lr" + formatted_lr
#
#     if args.epochs != config.EPOCHS:
#         name += "_ep" + str(args.epochs)
#
#     if args.memo is not None:
#         name += args.memo
#     return name

def generate_dapt_ckp_name(args):
    name = "dapt"

    # *** 关键修改：使用 format_lr_name 函数处理学习率 ***
    formatted_lr = format_lr_name(args.lr)
    name += "_lr" + formatted_lr

    name += "_ep" + str(args.epochs)

    if args.memo is not None:
        name += "_" + args.memo
    return name

def modify_if_exists(original_output_path):
    final_output_path = original_output_path
    i = 1
    # 检查路径是否存在，如果存在，则循环添加后缀
    while final_output_path.exists():
        print(f"目标目录已存在: {final_output_path}")
        new_dirname = f"{original_output_path.name}_{i}"
        # 构造新的完整路径：父目录 / 新目录名
        final_output_path = original_output_path.parent / new_dirname
        i += 1
    return final_output_path