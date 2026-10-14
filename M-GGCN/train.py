# 文件名: train_test.py
import os
import glob
import random
import warnings

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from model import Model
from feature_extractor import Gravity3DFeatureExtractor
from data_loader import load_graph_only, load_labels_only
from utils import find_label_files as original_find_label_files, save_model

from config import (
    DEVICE, MODEL_SAVE_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    S_NEIGHBORS, MAX_HOPS, VEC_DIM,
    HIDDEN_DIM, DROPOUT_P,
    LR, WEIGHT_DECAY, EPOCHS, LR_STEP_SIZE, LR_GAMMA, GRAD_CLIP_MAX_NORM, LOSS_ALPHA,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, LOSS_FIG_PATH, TAU_FIG_PATH, RESULT_CSV_PATH,
)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def find_label_files(network_path):
    if "verification_network" in network_path:
        base_name = os.path.splitext(os.path.basename(network_path))[0]
        label_dir = os.path.dirname(network_path).replace("verification_network", "verification_lables")
        return sorted(set(glob.glob(os.path.join(label_dir, f"{base_name}_*"))))
    return original_find_label_files(network_path)


def listmle_loss(output, labels, alpha=LOSS_ALPHA):
    out, lbl = output.float().view(-1), labels.float().view(-1)
    return ((1.0 + alpha * lbl) * (out - lbl) ** 2).mean()


def plot_history(data_dict, save_path, ylabel, title):
    plt.figure(figsize=(10, 5))
    for label, values in data_dict.items():
        if values:
            plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def _build_features(network_path, feature_extractor, VEC_DIM):
    """实时构建特征，不缓存"""
    G, adj, sorted_nodes = load_graph_only(network_path)
    if G is None:
        return None
    features_3d = feature_extractor.forward(G)
    if features_3d is None:
        return None

    N = features_3d.shape[0]
    features_2d = features_3d.view(N, VEC_DIM)

    if not torch.all(features_2d == 0):
        mean = features_2d.mean(dim=0, keepdim=True)
        std = features_2d.std(dim=0, keepdim=True)
        std[std < 1e-8] = 1.0
        features_2d = (features_2d - mean) / std

    return {"adj": adj.cpu(), "features": features_2d.cpu(), "sorted_nodes": sorted_nodes}


def prepare_data(network_paths, feature_extractor, VEC_DIM, data_dir="training"):
    all_data = []
    for network_path in tqdm(sorted(set(network_paths)), desc=f"处理 {data_dir} 数据集"):
        net_prefix = os.path.splitext(os.path.basename(network_path))[0]
        try:
            label_paths = find_label_files(network_path)
            if not label_paths:
                continue

            built = _build_features(network_path, feature_extractor, VEC_DIM)
            if built is None:
                continue

            item_base = {"adj": built["adj"], "features": built["features"]}
            for label_path in label_paths:
                labels, sub_name = load_labels_only(label_path, built["sorted_nodes"])
                if labels is not None:
                    all_data.append({**item_base, "name": f"{net_prefix}_{sub_name}", "labels": labels})
        except Exception:
            continue
    return all_data



def evaluate_dataset(model, dataset):
    model.eval()
    total_loss, total_tau, count = 0.0, 0.0, 0
    with torch.no_grad():
        for data in dataset:
            adj = data["adj"].to(DEVICE)
            features = data["features"].to(DEVICE)
            labels = data["labels"].to(DEVICE)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            for k in range(labels.shape[1]):
                if DEVICE.type == "cuda":
                    with torch.autocast(device_type=DEVICE.type):
                        output = model(features, adj)
                else:
                    output = model(features, adj)
                out_f = output.float().view(-1)
                lbl_k = labels[:, k].float().view(-1)
                total_loss += listmle_loss(out_f, lbl_k).item()
                out_np, lbl_np = out_f.cpu().numpy(), lbl_k.cpu().numpy()
                if len(out_np) > 1:
                    tau, _ = kendalltau(out_np, lbl_np)
                    if not np.isnan(tau):
                        total_tau += tau
                count += 1
    return (total_loss / count if count else 0.0), (total_tau / count if count else 0.0)


def train_model(model, train_data, val_data, optimizer, epochs, scheduler,
                model_init_params, best_model_path, final_model_path):
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "val_tau": []}
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else torch.amp.GradScaler('cpu')

    for epoch in range(epochs):
        model.train()
        total_loss, n_updates = 0, 0
        pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for data in pbar:
            adj = data["adj"].to(DEVICE)
            features = data["features"].to(DEVICE)
            labels = data["labels"].to(DEVICE)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            for k in range(labels.shape[1]):
                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type):
                    output = model(features, adj)
                out_f = output.float().view(-1)
                lbl_k = labels[:, k].float().view(-1)
                loss = listmle_loss(out_f, lbl_k)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_updates += 1

        avg_train = total_loss / n_updates if n_updates else 0
        history["train_loss"].append(avg_train)

        if val_data:
            avg_val, avg_tau = evaluate_dataset(model, val_data)
            history["val_loss"].append(avg_val)
            history["val_tau"].append(avg_tau)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_model(model, best_model_path, model_params=model_init_params)

        scheduler.step()

    save_model(model, final_model_path, model_params=model_init_params)
    return best_val_loss, history



def train_main():
    extractor = Gravity3DFeatureExtractor(s_neighbors=S_NEIGHBORS, max_hops=MAX_HOPS)

    def _glob_files(dir_path):
        files = []
        for pattern in ["**/*.txt", "**/*.csv"]:
            files.extend(glob.glob(os.path.join(dir_path, pattern), recursive=True))
        return sorted(set(files))

    train_files = _glob_files(TRAIN_DIR)
    if not train_files:
        return
    train_data = prepare_data(train_files, extractor, VEC_DIM, "training")
    random.shuffle(train_data)

    val_files = _glob_files(VAL_DIR)
    val_data = prepare_data(val_files, extractor, VEC_DIM, "validation") if val_files else []

    model_params = {"vec_dim": VEC_DIM, "hidden_dim": HIDDEN_DIM, "dropout_p": DROPOUT_P}
    model = Model(**model_params).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    _, history = train_model(model, train_data, val_data, optimizer, EPOCHS,
                             scheduler, model_params, BEST_MODEL_PATH, FINAL_MODEL_PATH)
    plot_history({"Training Loss": history["train_loss"], "Validation Loss": history["val_loss"]},
                 LOSS_FIG_PATH, "Loss", "Loss History")
    plot_history({"Validation Kendall Tau": history["val_tau"]},
                 TAU_FIG_PATH, "Tau", "Kendall Tau History")



def test_main():
    if not os.path.exists(BEST_MODEL_PATH):
        return None

    try:
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        init_params = ckpt.get('model_params', {"vec_dim": VEC_DIM, "hidden_dim": HIDDEN_DIM, "dropout_p": DROPOUT_P})
        init_params['vec_dim'] = VEC_DIM
        model = Model(**init_params).to(DEVICE)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model.eval()
    except Exception:
        return None

    extractor = Gravity3DFeatureExtractor(s_neighbors=S_NEIGHBORS, max_hops=MAX_HOPS)
    network_files = []
    for p in [os.path.join(TEST_DIR, "out.*"), os.path.join(TEST_DIR, "*.txt")]:
        network_files.extend(glob.glob(p))
    network_files = sorted(set(network_files))
    if not network_files:
        return None

    results = []

    for network_path in network_files:
        net_name = os.path.basename(network_path)
        label_paths = original_find_label_files(network_path)
        if not label_paths:
            continue

        built = _build_features(network_path, extractor, VEC_DIM)
        if built is None:
            continue
        sorted_nodes = built["sorted_nodes"]

        for label_path in label_paths:
            try:
                labels, _ = load_labels_only(label_path, sorted_nodes)
                if labels is None:
                    continue

                with torch.no_grad():
                    adj = built["adj"].to(DEVICE)
                    features = built["features"].to(DEVICE)
                    output = model(features, adj)
                    out_np = output.squeeze().cpu().numpy()
                    lbl_np = labels.squeeze().cpu().numpy()

                tau = 0.0
                if out_np.ndim > 0 and len(out_np) > 1:
                    try:
                        tau, _ = kendalltau(out_np, lbl_np)
                    except Exception:
                        pass
                    if np.isnan(tau):
                        tau = 0.0

                results.append({"Network": net_name,
                                "Infection_Rate_File": os.path.basename(label_path),
                                "Tau": tau})
            except Exception:
                continue

    if not results:
        return None

    df = pd.DataFrame(results)
    df.to_csv(RESULT_CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train_main()
    if args.mode in ["test", "both"]:
        test_main()