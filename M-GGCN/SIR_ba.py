from SIR_real import *


if __name__ == '__main__':
    adj_path = r'./data/ws1'
    labels_path = r'./data/ws2'
    work = os.listdir(adj_path)
    print(work)

    n = 500
    for w in work:
        print(f'当前w：{w}')
        start_time = time.time()
        adjacency_matrix = pd.read_csv(os.path.join(adj_path, w), header=None).to_numpy()
        G = nx.from_numpy_matrix(adjacency_matrix)  # 构建图
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

        # beta2 = [beta * i / 10 for i in range(10, 31, 5)]
        beta2 = [beta * 1.0]
        print(beta2)
        for infe_rate in beta2:
            # 开始循环
            print(f'当前rate:{infe_rate}')
            for i in range(10):
                output = SIR_ALL_NODES(G, infe_rate, n)
                # print(output)
                # 写入文件
                with open(os.path.join(labels_path, w[:-4] + '_' + "{:.1f}".format(infe_rate/beta) + f'_{i}.csv'), 'w') as f:      # todo：要求真实标签的文件要为y_adj
                    for k in sorted(output, key=output.__getitem__, reverse=True):
                        f.write("%s, %s\n" % (k, output[k]))

            end_time = time.time()
            run_time = end_time - start_time
            print("程序运行时间：", run_time, "秒")


