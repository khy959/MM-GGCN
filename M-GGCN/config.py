# 文件名: config.py
import os
import torch


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"[Config] 已启用CUDA，使用GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[Config] 未检测到CUDA设备，将使用CPU。")



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")

TRAIN_DIR = os.path.join(PROJECT_ROOT, "Networks", "training")
VAL_DIR = os.path.join(PROJECT_ROOT, "verification_network")
TEST_DIR = os.path.join(PROJECT_ROOT, "Networks", "real")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)



S_NEIGHBORS = 32
MAX_HOPS = 3
VEC_DIM = S_NEIGHBORS * MAX_HOPS


# ============================== 模型结构 ==============================
HIDDEN_DIM = 128
DROPOUT_P = 0.8


BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
LOSS_FIG_PATH = os.path.join(MODEL_SAVE_DIR, "loss_history.png")
TAU_FIG_PATH = os.path.join(MODEL_SAVE_DIR, "tau_history.png")
RESULT_CSV_PATH = os.path.join(PROJECT_ROOT, "real_networks_evaluation.csv")