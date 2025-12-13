from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- 数据路径配置（使用绝对路径）---
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA = PROCESSED_DATA_DIR / "processed_data.csv"
INITIAL_URLS = RAW_DATA_DIR / "urls_initial.csv"
CRAWLED_DATA = PROCESSED_DATA / "crawled_data.csv"

# --- 模型配置（使用绝对路径）---
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_WEIGHTS_DIR = MODELS_DIR / "weights"
MODELS_LOG_DIR =  MODELS_DIR / "logs"
MODELS_BASE_CPS = MODELS_DIR / "base_checkpoints"
MODELS_DAPT_CPS = MODELS_DIR / "dapt_checkpoints"

BEST_MODEL_WEIGHTS = MODELS_WEIGHTS_DIR / "model.pt"

# --- 报告和图表路径 ---
REPORT_PATH = "final_report.pdf"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ---
BERT = "distilbert-base-uncased"
