import math
import igraph as ig
import networkx as nx
import numpy as np
import torch


def _count_common_neighbors(u, v, adj_indptr, adj_indices):
    start_u, end_u = adj_indptr[u], adj_indptr[u + 1]
    start_v, end_v = adj_indptr[v], adj_indptr[v + 1]
    count = 0
    idx_u, idx_v = start_u, start_v
    while idx_u < end_u and idx_v < end_v:
        nu, nv = adj_indices[idx_u], adj_indices[idx_v]
        if nu == nv:
            count += 1
            idx_u += 1
            idx_v += 1
        elif nu < nv:
            idx_u += 1
        else:
            idx_v += 1
    return count


def _compute_sim_vectors(adj_indptr, adj_indices, ks_arr, deg_arr, s_neighbors):
    num_nodes = len(ks_arr)
    sim_vecs = np.zeros((num_nodes, 6), dtype=np.float32)

    for i in range(num_nodes):
        ks_i, deg_i = ks_arr[i], deg_arr[i]
        queue = np.zeros(num_nodes, dtype=np.int32)
        distances = np.full(num_nodes, -1, dtype=np.int32)
        visited = np.zeros(num_nodes, dtype=bool)
        queue[0], visited[i], distances[i] = i, True, 0
        head, tail, neighbor_count = 0, 1, 0
        c_ks_gt = c_ks_eq = c_ks_lt = 0.0
        c_deg_gt = c_deg_eq = c_deg_lt = 0.0

        while head < tail and neighbor_count < s_neighbors:
            u = queue[head]
            head += 1
            if u != i:
                neighbor_count += 1
                if ks_arr[u] > ks_i:
                    c_ks_gt += 1.0
                elif ks_arr[u] == ks_i:
                    c_ks_eq += 1.0
                else:
                    c_ks_lt += 1.0

                if deg_arr[u] > deg_i:
                    c_deg_gt += 1.0
                elif deg_arr[u] == deg_i:
                    c_deg_eq += 1.0
                else:
                    c_deg_lt += 1.0

            start_ptr, end_ptr = adj_indptr[u], adj_indptr[u + 1]
            for idx in range(start_ptr, end_ptr):
                v = adj_indices[idx]
                if not visited[v]:
                    visited[v] = True
                    distances[v] = distances[u] + 1
                    queue[tail] = v
                    tail += 1

        sim_vecs[i, 0], sim_vecs[i, 1], sim_vecs[i, 2] = c_ks_gt, c_ks_eq, c_ks_lt
        sim_vecs[i, 3], sim_vecs[i, 4], sim_vecs[i, 5] = c_deg_gt, c_deg_eq, c_deg_lt

    return sim_vecs


def _cosine_similarity(vec_a, vec_b):
    dot = na = nb = 0.0
    for k in range(6):
        dot += vec_a[k] * vec_b[k]
        na += vec_a[k] ** 2
        nb += vec_b[k] ** 2
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9)


def _compute_features(
        adj_indptr, adj_indices, ks_arr, deg_arr, sim_vecs,
        s_neighbors, max_hops, beta, zeta
):

    num_nodes = len(ks_arr)
    node_matrices = np.zeros((num_nodes, s_neighbors, max_hops), dtype=np.float32)

    for i in range(num_nodes):
        deg_i, ks_i = deg_arr[i], ks_arr[i]
        sim_vec_i = sim_vecs[i]

        queue = np.zeros(num_nodes, dtype=np.int32)
        distances = np.full(num_nodes, -1, dtype=np.int32)
        queue[0], distances[i] = i, 0
        head, tail, collected = 0, 1, 0
        list_n, list_d = [], []

        while head < tail:
            u = queue[head]
            head += 1
            if u != i and collected < s_neighbors:
                list_n.append(u)
                list_d.append(distances[u])
                collected += 1
            if distances[u] >= max_hops:
                continue
            start, end = adj_indptr[u], adj_indptr[u + 1]
            for idx in range(start, end):
                v = adj_indices[idx]
                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    queue[tail] = v
                    tail += 1

        while len(list_n) < s_neighbors:
            list_n.append(-1)
            list_d.append(-1)

        arr_n = np.array(list_n, dtype=np.int32)
        arr_d = np.array(list_d, dtype=np.int32)
        valid_mask = arr_d != -1
        if np.any(valid_mask):
            sorted_indices = np.argsort(arr_d[valid_mask])
            arr_n[:len(sorted_indices)] = arr_n[valid_mask][sorted_indices]
            arr_d[:len(sorted_indices)] = arr_d[valid_mask][sorted_indices]

        for s_idx in range(s_neighbors):
            j, d_ij = arr_n[s_idx], arr_d[s_idx]
            if j == -1 or d_ij <= 0:
                continue

            deg_j = deg_arr[j]
            deg_sum = deg_i + deg_j

            s_struct = (
                _count_common_neighbors(i, j, adj_indptr, adj_indices) / deg_sum
                if deg_sum > 0 else 0.0
            )
            s_attr = _cosine_similarity(sim_vec_i, sim_vecs[j])
            interaction_ij = beta * s_struct + (1.0 - beta) * s_attr

            mi_j = deg_i + zeta * (deg_i * deg_j * interaction_ij)
            mj_i = deg_j + zeta * (deg_j * deg_i * interaction_ij)

            F_ij = (mi_j * mj_i) / float(d_ij ** 2)
            node_matrices[i, s_idx, d_ij - 1] = F_ij

    return node_matrices



class Gravity3DFeatureExtractor:
    def __init__(self, s_neighbors=32, max_hops=3, beta=0.5, zeta=1.0):
        self.s_neighbors = s_neighbors
        self.max_hops = max_hops
        self.beta = beta
        self.zeta = zeta

    def forward(self, G):
        if isinstance(G, ig.Graph):
            adj_scipy = G.get_adjacency_sparse().tocsr()
            adj_scipy.sort_indices()
            deg_arr = np.array(G.degree(), dtype=np.float32)
            ks_arr = np.array(G.coreness(), dtype=np.float32)

        elif isinstance(G, nx.Graph):
            nodes = sorted(G.nodes())
            num_nodes = len(nodes)
            adj_scipy = nx.adjacency_matrix(G, nodelist=nodes).tocsr()
            adj_scipy.sort_indices()

            src, dst = adj_scipy.nonzero()
            mask = src < dst
            edges = np.column_stack((src[mask], dst[mask]))

            g_ig = ig.Graph(n=num_nodes, edges=edges, directed=False)
            deg_arr = np.array(g_ig.degree(), dtype=np.float32)
            ks_arr = np.array(g_ig.coreness(), dtype=np.float32)
        else:
            raise TypeError(f"不支持的图格式: {type(G)}")

        adj_indices = adj_scipy.indices
        adj_indptr = adj_scipy.indptr


        sim_vecs = _compute_sim_vectors(
            adj_indptr, adj_indices, ks_arr, deg_arr, self.s_neighbors
        )


        node_matrices = _compute_3d_gravity_features(
            adj_indptr, adj_indices, ks_arr, deg_arr, sim_vecs,
            self.s_neighbors, self.max_hops, self.beta, self.zeta
        )

        features_3d = torch.tensor(node_matrices, dtype=torch.float32)
        features_3d = torch.log1p(features_3d)

        return features_3d