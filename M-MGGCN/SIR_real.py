import networkx as nx
import random
import time
import multiprocessing as mp
import pandas as pd
import os
import numpy as np


def load_graph(path):
    G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


def SIR_EACH_NODE(G, infected, INFE_RATE, n):
    re = 0      # 恢复人数，作为影响力
    # n = 100    # 运行次数
    # print(f'当前节点：{infected[0]}')
    for _ in range(n):
        INFE = set(infected)
        R = set()
        while len(INFE) != 0:
            newInf = []
            for i in INFE:
                for j in G.neighbors(i):
                    k = random.uniform(0, 1)
                    if (INFE_RATE > k) and (j not in INFE) and (j not in R):
                        newInf.append(j)
                R.add(i)
            INFE = set(newInf)
        re += len(R)
    return(re/n), infected[0]


def SIR_ALL_NODES(G, INFE_RATE, n):
    node_list = list(G.nodes())
    SIR_dict = {}
    pool = mp.Pool(processes=10)
    results = []
    for node in node_list:
        res = pool.apply_async(SIR_EACH_NODE, [G, [node], INFE_RATE, n])
        results.append(res)
#         sir = SIR_EACH_NODE(G,[node],INFE_RATE,RECO_RATE)
#         SIR_dict[node] = sir
    pool.close()
    pool.join()
    for r in results:
        sir, node = r.get()
        SIR_dict[node] = sir
    # print('存储字典:\n',SIR_dict)
    return SIR_dict


def cal_beta(adj):
    # 计算传播阈值，adj为torch类型
    deg = adj.sum(1)
    deg2 = deg**2
    m_deg = np.mean(deg)
    m_deg2 = np.mean(deg2)
    beta = m_deg / (m_deg2 - m_deg)
    print(beta)
    return max(np.round(beta, 4), 0.01)


def cal_connetion_graph(g):
    is_connected = nx.is_connected(g)
    if is_connected:
        return g
    else:
        # 获取所有连通分量
        components = list(nx.connected_components(g))
        # 找到具有最多节点的连通分量
        largest_component = max(components, key=len)
        # 创建最大全连通子图
        subgraph = g.subgraph(largest_component)
        return subgraph


def cal_b(beta):
    # 输入转播阈值，输出阈值最近的十个传播概率
    # 生成前五个0.01的数
    before_nums = [beta - i * 0.01 for i in range(5)]
    # 删除前五个数中小于等于0的数
    before_nums = [n for n in before_nums if n > 0]
    # 计算需要补充的数个数
    missing_count = 5 - len(before_nums)
    # 生成后五个0.01的数
    after_nums = [beta + i * 0.01 for i in range(1, 6 + missing_count)]
    before_nums.reverse()
    before_nums.extend(after_nums)
    return before_nums



if __name__ == '__main__':
    path = r'./data/real_network'
    label_path = r'./data/test100'
    work2 = os.listdir(path)
    n = 200
    print(work2)
    for w in work2:
        print(f'当前w：{w}')
        start_time = time.time()
        G = nx.read_edgelist(os.path.join(path, w))
        # 判断是否全连通
        G = cal_connetion_graph(G)
        print("节点数量:", G.number_of_nodes())
        print("边数量:", G.number_of_edges())
        is_directed = nx.is_directed(G)
        if is_directed:
            continue

        # 计算传播阈值
        beta = cal_beta(nx.to_numpy_array(G))

        # beta2 = np.array(cal_b(beta))
        # beta2 = beta2[beta2 > 0]

        beta2 = [beta * i/10 for i in range(10, 31, 5)]
        print(beta2)
        for infe_rate in beta2:
        # for infe_rate in [0.10]:
            # 开始循环
            print(f'当前rate:{infe_rate}')
            for i in range(50):
                output = SIR_ALL_NODES(G, infe_rate, n)
                # print(output)
                # 写入文件
                with open(os.path.join(label_path, w[4:] + '_' + "{:.1f}_".format(infe_rate/beta) + f'{i}'.zfill(2) + '.csv'), 'w') as f:
                    for k in sorted(output, key=output.__getitem__, reverse=True):
                        f.write("%s, %s\n" % (k, output[k]))

            end_time = time.time()
            run_time = end_time - start_time
            print("程序运行时间：", round(run_time, 4), "秒")

