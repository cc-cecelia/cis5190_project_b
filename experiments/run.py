import subprocess
import os
import sys
import time
import datetime


def run_command(command, log_name):
    """
    执行命令，实时打印输出并保存到 logs 文件夹。
    """
    log_dir = "../models/temp_logs/bash"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = f"{log_dir}/{log_name}_{timestamp}.log"

    print(f"\n{'=' * 80}")
    print(f"🚀 [START] {log_name}")
    print(f"📄 Log: {log_path}")
    print(f"⌨️  Cmd: {command}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    my_env = os.environ.copy()
    my_env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as f:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 把错误也重定向到标准输出，防止错位
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=my_env
        )

        # 【关键修改 3】使用 readline() 循环读取，并手动 flush
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # 打印到屏幕并强制刷新
                sys.stdout.write(line)
                sys.stdout.flush()
                # 写入文件并强制刷新
                f.write(line)
                f.flush()

    duration = (time.time() - start_time) / 60

    if process.returncode != 0:
        print(f"\n❌ [FAILED] {log_name} (Duration: {duration:.2f} min)")
        print(f"Check log file: {log_path}")
        # 如果 DAPT 挂了，后面依赖它的实验也会挂，所以直接退出比较安全
        if "dapt" in log_name:
            print("CRITICAL: DAPT phase failed. Aborting subsequent experiments.")
            sys.exit(1)
    else:
        print(f"\n✅ [SUCCESS] {log_name} (Duration: {duration:.2f} min)")


def main():
    # 检查当前目录
    if os.path.basename(os.getcwd()) != "src":
        print("⚠️  请在 'src' 目录下运行此脚本！")
        sys.exit(1)

    print("🛌 启动全自动 Ablation Study (Two-Stage Epochs Supported) 流程...")
    total_start = time.time()

    # ==========================================
    # 1. Baseline: Logistic Regression
    # ==========================================
    # run_command(
    #     "python baseline/train_baseline.py --max_features 5000 --ngram_range 1 2 --C 1.0",
    #     "00_baseline_logistic"
    # )

    # ==========================================
    # 2. Exp 1: DistilBERT (Frozen)
    # ==========================================
    # 策略：冻结 Encoder，只训练 Classifier
    # 注意：Frozen 模式下，warmup 参数必须传
    # for batch_size in [8, 16, 32, 64]:
    #     for dropout in [0.1, 0.3, 0.5]:
    #         run_command(
    #             "python train.py --freeze_encoder --warm_lr 2e-5 --batch_size 16 --warm_ep 5 --dropout 0.1 --memo \"15k\"",
    #             f"01_exp1_frozen_baseline_bs{batch_size}_do{str(int(dropout*10))}"
    #         )

    # best_batch = 16 #
    # best_dropout = 0.1 #

    # ==========================================
    # 3. DAPT Phase (Pre-training)
    # ==========================================
    # 这是 Exp 2 和 Exp 4 的前置条件 DONE
    run_command(
        "python train_dapt.py --lr 3e-5 --batch_size 16 --epochs 3 --memo \"large\" --csv_name processed_data_large.csv",
        "02_dapt_pretraining"
    )

    # ==========================================
    # 4. Exp 2: DistilBERT + DAPT (Frozen)
    # ==========================================
    # 策略：加载 DAPT 权重，但依然冻结 Encoder
    # for warm_lr in [1e-5, 2e-5, 3e-5]:
    #     run_command(
    #         f"python train.py --use_dapt --checkpoint dapt_lr3e5_ep3_15k_backbone --freeze_encoder --warm_lr {warm_lr} --batch_size {best_batch} --warm_ep 5 --dropout {best_dropout} --memo \"15k\"",
    #         f"03_exp2_dapt_frozen_{warm_lr}"
    #     )

    # ==========================================
    # 5. Exp 3: DistilBERT (Two-Stage Fine-tuning)
    # ==========================================
    # 策略：两阶段微调 (不加 DAPT)
    # 阶段1: 3轮 Warmup (lr=1e-4)
    # 阶段2: 5轮 Full FT (lr=2e-5)
    # run_command(
    #     "python train.py --bert_lr 2e-5 --warm_lr 1e-4 --warm_ep 3 --bert_ep 5 --batch_size 16 --dropout 0.1 --memo \"3k8\"",
    #     "04_exp3_finetune_twostage"
    # )

    # ==========================================
    # 6. Exp 4: DistilBERT + DAPT (Two-Stage Fine-tuning)
    # ==========================================
    # 策略：加载 DAPT 权重 + 两阶段微调
    # 阶段1: 3轮 Warmup (lr=1e-4)
    # 阶段2: 5轮 Full FT (lr=2e-5)
    run_command(
        "python train.py --use_dapt --checkpoint dapt_lr3e05_ep3_large_backbone --bert_lr 2e-5 --warm_lr 1e-4 --warm_ep 3 --bert_ep 5 --batch_size 16 --dropout 0.1  --memo \"large\" --csv_name processed_data_large.csv",
        "05_exp4_dapt_finetune_twostage"
    )

    total_duration = (time.time() - total_start) / 60
    print(f"\n🎉🎉🎉 所有任务执行完毕！大家晚安！")
    print(f"总耗时: {total_duration:.2f} 分钟")


if __name__ == "__main__":
    main()