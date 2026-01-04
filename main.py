import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.utils.data_loader import prepare_data
from src.engine.trainer import train_v11
from src.engine.evaluator import evaluate_model

# 路徑設定
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "USD_TWD.csv"
SAVE_PATH = BASE_DIR / "results" / "main"

# 硬體與參數設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
HORIZON = 3
LOOKBACK = 30
NUM_EPOCHS = 120
LR = 0.001

# 確保結果資料夾存在
if not SAVE_PATH.exists():
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

# 設定隨機種子 (Reproducibility)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[Setup] Device: {DEVICE} | Seed: {SEED} | Horizon: {HORIZON}")

# 3. Main Execution Flow: 執行流程
if __name__ == "__main__":

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        exit()

    print("Loading Data...")
    df = pd.read_csv(DATASET_PATH)

    train_loader, test_loader, scalers_raw, _, _, _, _ = prepare_data(
        df, lookback=LOOKBACK, horizon=HORIZON
    )

    model = train_v11(
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        horizon=HORIZON,
        num_epochs=NUM_EPOCHS,
        lr=LR
    )

    print("Evaluating Model...")

    evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler=scalers_raw['target'],
        device=DEVICE,
        horizon=HORIZON
    )

    print("\nAll tasks completed.")
