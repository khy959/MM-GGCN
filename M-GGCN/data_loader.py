
import os
import logging

import igraph as ig
import numpy as np
import torch

log = logging.getLogger(__name__)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    if adj.device.type != "cpu":
        adj = adj.cpu()

    adj_selfloop = adj + torch.eye(adj.shape[0], device="cpu")
    row_sum = adj_selfloop.sum(1, keepdim=True)
    row_sum[row_sum == 0] = 1

    return adj_selfloop / row_sum



def load_graph_only(network_path):
    try:
        # ---------- 1. 读取图 ----------
        if network_path.endswith(".csv"):
            adj_matrix_np = np.loadtxt(network_path, delimiter=",")
            G_orig = ig.Graph.Weighted_Adjacency(
                adj_matrix_np.tolist(), mode="undirected", loops=False
            )
            G_orig.vs["name"] = list(range(G_orig.vcount()))
        else:
            edges = []
            with open(network_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("%", "#")):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        edges.append((int(parts[0]), int(parts[1])))

            if not edges:
                return None, None, None

            unique_nodes = list({u for e in edges for u in e})
            name2vid = {name: i for i, name in enumerate(unique_nodes)}

            G_orig = ig.Graph()
            G_orig.add_vertices(len(unique_nodes))
            G_orig.vs["name"] = unique_nodes
            G_orig.add_edges([(name2vid[u], name2vid[v]) for u, v in edges])

        if G_orig.vcount() == 0:
            return None, None, None

        # ---------- 2. 清洗 + LCC ----------
        G_orig.simplify(multiple=True, loops=True)

        if not G_orig.is_connected():
            G_processed = G_orig.components().giant()
        else:
            G_processed = G_orig.copy()

        if G_processed.vcount() == 0:
            return None, None, None


        current_names = G_processed.vs["name"]
        sorted_vids = sorted(
            range(G_processed.vcount()), key=lambda i: current_names[i]
        )
        G_final = G_processed.permute_vertices(sorted_vids)
        sorted_nodes = G_final.vs["name"]


        adj_dense = G_final.get_adjacency_sparse().toarray().astype(np.float32)
        adj_normalized = normalize_adj(torch.from_numpy(adj_dense))

        return G_final, adj_normalized, sorted_nodes

    except Exception as e:
        log.error(f"加载图 {network_path} 失败: {e}", exc_info=True)
        return None, None, None



def load_labels_only(label_path, sorted_nodes, mode="log_div_mean"):

    try:
        num_nodes = len(sorted_nodes)

        try:
            labels_df = np.loadtxt(label_path, delimiter=",", skiprows=1)
        except Exception:
            labels_df = np.loadtxt(label_path, delimiter=",")

        if labels_df.ndim == 1:
            labels_df = np.expand_dims(labels_df, axis=0)


        labels_dict = {int(row[0]): float(row[1]) for row in labels_df}

        aligned_labels = np.zeros(num_nodes, dtype=np.float32)
        for idx, node_id in enumerate(sorted_nodes):
            aligned_labels[idx] = labels_dict.get(node_id, 0.0)

        if mode == "sir_div_n":
            normalized_labels = aligned_labels / float(num_nodes)
        labels_tensor = torch.from_numpy(normalized_labels).float().view(-1, 1)
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        return labels_tensor, label_name

    except Exception as e:
        log.error(f"加载标签 {label_path} 失败: {e}")
        return None, None