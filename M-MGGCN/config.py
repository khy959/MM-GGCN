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
FEATURE_CACHE_DIR = os.path.join(PROJECT_ROOT, "feature_cache")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)



TRAIN_DIR = os.path.join(PROJECT_ROOT, "Networks", "training")
VAL_DIR = os.path.join(PROJECT_ROOT, "verification_network")
TEST_DIR = os.path.join(PROJECT_ROOT, "Networks", "real")
CACHE_DIR = FEATURE_CACHE_DIR



S_NEIGHBORS = 32
MAX_HOPS = 3
VEC_DIM = S_NEIGHBORS * MAX_HOPS


HIDDEN_DIM = 128


BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
LOSS_FIG_PATH = os.path.join(MODEL_SAVE_DIR, "loss_history.png")
TAU_FIG_PATH = os.path.join(MODEL_SAVE_DIR, "tau_history.png")
RESULT_CSV_PATH = os.path.join(PROJECT_ROOT, "real_networks_evaluation.csv")