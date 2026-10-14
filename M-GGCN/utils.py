

import os
import glob
import logging

import torch

try:
    from config import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from model import Model
except ImportError:
    Model = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ===================================================================
# 1. 查找标签文件
# ===================================================================
def find_label_files(network_path):
    """
    查找与给定网络文件对应的标签文件。
    支持 'SIR_results' 和 'real_results' 两个目录，
    并支持 'out.xxxx' 格式文件名的智能匹配。
    """
    filename = os.path.basename(network_path)

    # 智能提取文件名核心
    if filename.startswith("out."):
        file_stem = filename
        alternative_stem = filename[4:]
    else:
        file_stem = os.path.splitext(filename)[0]
        alternative_stem = None

    parent_folder_name = os.path.basename(os.path.dirname(network_path))
    possible_base_dirs = ["SIR_results", "real_results"]
    found_label_paths = []

    for base_label_dir in possible_base_dirs:
        if not os.path.exists(base_label_dir):
            continue

        search_dirs = [
            os.path.join(base_label_dir, file_stem),
            os.path.join(base_label_dir, parent_folder_name),
            base_label_dir,
            os.path.dirname(network_path),
        ]
        if alternative_stem:
            search_dirs.insert(1, os.path.join(base_label_dir, alternative_stem))
        search_dirs = sorted(set(search_dirs))

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue

            patterns = [
                os.path.join(search_dir, f"{file_stem}*.txt"),
                os.path.join(search_dir, f"{file_stem}*.csv"),
            ]
            if alternative_stem:
                patterns += [
                    os.path.join(search_dir, f"{alternative_stem}*.txt"),
                    os.path.join(search_dir, f"{alternative_stem}*.csv"),
                ]

            for pattern in patterns:
                for f in glob.glob(pattern):
                    if os.path.abspath(f) != os.path.abspath(network_path):
                        found_label_paths.append(f)

    return sorted(set(found_label_paths))


# ===================================================================
# 2. 模型保存与加载
# ===================================================================
def save_model(model, path, model_params=None):
    """保存模型状态和初始化参数"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_params": model_params,
    }
    torch.save(checkpoint, path)
    log.info(f"模型已保存至: {path}")


def load_model(path, device=DEVICE, model_class=Model):
    """加载模型"""
    if not os.path.exists(path):
        log.error(f"模型文件不存在: {path}")
        return None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if "model_params" in checkpoint and checkpoint["model_params"] is not None:
            model_args = checkpoint["model_params"]
        else:
            log.error("Checkpoint 中未找到 'model_params'，无法重建模型结构。")
            return None

        model = model_class(**model_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        log.info(f"成功加载模型: {path}")
        return model
    except Exception as e:
        log.error(f"加载模型 checkpoint 失败: {e}", exc_info=True)
        return None