import math
import time
import os
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def ResolveGraphFile(graph_path, weighted=True, sep='\t'):
    with open(graph_path) as graph_file:
        s = graph_file.readlines()
    adj_table = {}
    no_line = 0
    for line in s:
        no_line += 1
        temp = line.strip().split(sep)
        if weighted:
            weight = float(temp[2])
        else:
            weight = 1

        if temp[0] not in adj_table:
            adj_table.update({temp[0]: {temp[1]: weight}})
        elif temp[1] not in adj_table[temp[0]]:
            adj_table[temp[0]].update({temp[1]: weight})
        else:
            adj_table[temp[0]][temp[1]] += weight

        if temp[1] not in adj_table:
            adj_table.update({temp[1]:{temp[0]: weight}})
        elif temp[0] not in adj_table[temp[1]]:
            adj_table[temp[1]].update({temp[0]: weight})
        else:
            adj_table[temp[1]][temp[0]] += weight
    # print("number of vertices: ", len(adj_table))
    # print("number of edges: ", no_line)
    return adj_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.001)
    # 44 embedding files in total, so the embeddingPtr should in [0,43]
    parser.add_argument("--embeddingPtr", type=int, default=0)
    parser.add_argument("--maxCommunitySize", type=int, default=3000)
    args = parser.parse_args()

    start = '53858'
    k = args.maxCommunitySize
    print("k = {}".format(k))
    # grountruth
    with open("gt_20220722.txt") as f:
        gt = eval(f.readline())
        

    graphFile = "combined_20220722.gml"
    entropyCacheFile = "entropyEmbeddingCacheFile.npz"

    # if(not os.path.exists(entropyCacheFile)):
    #     vec_dicts = []
    #     vec_file_name = []
    #     import os
    #     # path = "Tsinghua embedding"
    #     path = "./"
    #     vec_files= os.listdir(path)
    #     for file in vec_files:
    #         if(not file.endswith(".emb")):
    #             continue
    #         vec_dict = {}
    #         f = open(path + "/" + file)
    #         for line in f:
    #             node = line.split(":")[0]
    #             vec = np.array(list(map(float, line.split(":")[1].split())))
    #             vec_dict[node] = vec
    #         vec_dicts.append(vec_dict)
    #         vec_file_name.append(file)
            

    #     graph_path = "combined_20220722.txt"
    #     # 调用函数将存储图的文件解析成用字典存的邻接表，用self.adj表示图的邻接表
    #     adjacency_table = ResolveGraphFile(graph_path, weighted=True, sep=" ")

    #     print("vec_dicts length:", len(vec_dicts))
    #     print("vec_file_name length:", len(vec_file_name))
    #     print("adjacency_table shape:", len(adjacency_table))

    #     np.savez(entropyCacheFile,vec_dicts=vec_dicts,vec_file_name=vec_file_name,adjacency_table=adjacency_table)
    # else:
    npzfile=np.load(entropyCacheFile, allow_pickle=True)
    vec_dicts = npzfile['vec_dicts']
    vec_file_name = npzfile['vec_file_name']
    adjacency_table = npzfile['adjacency_table'][()]
    print("vec_dicts length:", len(vec_dicts))
    print("vec_file_name length:", len(vec_file_name))
    print("adjacency_table shape:", len(adjacency_table))





    DG = nx.read_gml(graphFile)
    pi = nx.pagerank(DG)

    # for u, v in DG.edges:
    #         DG[u][v]['weight'] = float(DG[u][v]['weight'])
            
    # with open("combined_20220722.txt","w") as f:
    #     for u, v in DG.edges:
    #         DG[u][v]['weight'] = float(DG[u][v]['weight'])
    #         f.write("{} {} {}\n".format(u, v, float(DG[u][v]['weight'])))
            
            


    # 用字典存储每个节点的度数
    degree = {}
    # 存储所有度数的和
    m = 0
    # 计算每个节点的度数，以及所有节点度数之和
    for node in adjacency_table.keys():
        degree[node] = 0
        for neighbor, deg in adjacency_table[node].items():
            m += deg
            degree[node] += deg
    stopRes = []
    maxF1Res = []
    for beta in range(1000,100,-100):
        beta *= 1e-3
        # beta = args.beta
        stopRes.append([])
        maxF1Res.append([])
        # for embedding in range(len(vec_dicts)):
        embedding = args.embeddingPtr
        betaMsg = "-----------------beta" + str(beta) + "--embedding " + vec_file_name[embedding] + "-----------------"
        print(betaMsg)
        community = [start] # 存储社区节点的列表
        com_vec = vec_dicts[embedding][start]
        com_dis = dict()
        neighbors = {} # 当前社区的邻居字典集合
        vol1, g1 = 0.0, 0.0
        vol2, g2 = 0.0, 0.0
        vol3, g3 = 0.0, 0.0
        g12 = 0.0        # 计算当前社区的邻居字典集合以及当前社区的体积与割边
        for neighbor, deg in adjacency_table[start].items():
            neighbors.update({neighbor: deg})
            vol1 += deg
            g1 += deg
        delta = 0.0        # 初始化，计算合并前后的结构熵的变化值delta
        C = 0.0
        D = float("inf")
        for node in neighbors: # 对于邻居中的每一个节点，选出结构熵变化最大的邻居节点与当前社区合并
            vol3 = vol1 + degree[node] # 合并后的社区体积
            g12 = 0.0
            g2 = degree[node]
            neighbors_if_merge = {}
            for neighbor in adjacency_table[node]:
                if neighbor in community:
                    g12 += adjacency_table[node][neighbor]
                else:
                    neighbors_if_merge.update({neighbor:adjacency_table[node][neighbor]})
            g3 = g1 + g2 - 2 * g12            # 计算并存下合并后结构熵变化的值
            item1 = (1/m)*(vol1*math.log(vol1, 2) - vol3*math.log(vol3, 2) - (g1*math.log(vol1, 2) - g3*math.log(vol3,2)))
            item2 = (1/m)*(2*g12*math.log(m, 2))
            delta_if_merge = item1 + item2            # 存下结构熵变化最大的邻居节点

            # 计算node和start的欧式距离
            node_vec = vec_dicts[embedding][node]
            dis = np.linalg.norm(node_vec - com_vec)
            com_dis[node] = dis

            # 存下欧氏距离delta变化最大的邻居节点
            if -delta_if_merge * (1 - beta) + dis * beta < D:
                D = -delta_if_merge * (1 - beta) + dis * beta

            # if -delta_if_merge < delta:
                delta = -delta_if_merge
                com2 = [node]
                vol3_after_merge = vol3
                g3_after_merge = g3
                neighbors_extra = neighbors_if_merge
            

        cut = degree[start]
        phi = 1
        di_phi = 1
        PHI = [phi]
        vol = degree[start]


        # 当社区的大小小于参数 k 时 并且 delta 小于 0 ，也就是说结构熵仍然在减小，
        # 否则循环继续执行。
        # begin = time.time()
        deltaE = []
        deri_1 = [0]
        flag = 1
        # beta = 1e-2
        max_f1 = 0
        max_f1_precision = 0
        max_f1_recall_nodeNum = 0
        stop_f1 = 2/2691
        stop_nodenum = 1
        stop_precision = 1

        while len(community) < k:
            # 1. 合并delta最大的两个社区，以及更新相关的数据结构，包括社区列表，社区体积大小，
            # 割边数目，邻居集合等
            # 2. 继续尝试将社区的邻居加入到社区之中，并记下结构熵减小最大的邻居

            deltaE.append(delta)
            community += com2
            vol1 = vol3_after_merge
            g1 = g3_after_merge
            del neighbors[com2[0]]
            neighbors.update(neighbors_extra)

            # print(len(community), com2[0], int(com2[0]) // 2690, com2[0] in gt)

            com_vec = ((len(community)-1) * com_vec + vec_dicts[embedding][com2[0]]) / len(community)
            
            seq = community
            num_gt = len(gt)
            num_seq = len(seq)
            num_right = len(set(gt) & set(seq))
            
            recall_nodeNum = num_right / num_gt
            precision_nodeNum = num_right / num_seq
            f1_nodeNum = 2 * (precision_nodeNum * recall_nodeNum) / (precision_nodeNum + recall_nodeNum)
            if f1_nodeNum > max_f1:
                node_num = num_seq
                max_f1 = f1_nodeNum
                max_f1_precision = precision_nodeNum
                max_f1_recall_nodeNum = recall_nodeNum

            
            neighbors_if_merge = {}
            delta = 0.0
            C = 0.0
            D = float("inf")
            for node in neighbors:
                vol3 = vol1 + degree[node]
                g12 = 0.0
                g2 = degree[node]
                neighbors_if_merge = {}
                for neighbor in adjacency_table[node]:
                    if neighbor in community:
                        g12 += adjacency_table[node][neighbor]
                    else:
                        neighbors_if_merge.update({neighbor:adjacency_table[node][neighbor]})
                g3 = g1 + g2 - 2*g12
                # 计算并存下合并后结构熵变化的值
                item1 = (1/m)*(vol1*math.log(vol1, 2) - vol3*math.log(vol3, 2) - (g1*math.log(vol1, 2) - g3*math.log(vol3,2)))
                item2 = (1/m)*(2*g12*math.log(m, 2))
                delta_if_merge = item1 + item2

                # 计算node和当前community的欧式距离
                node_vec = vec_dicts[embedding][node]
                dis = np.linalg.norm(node_vec - com_vec)

                com_vec_tmp = vec_dicts[embedding][com2[0]]
                dis_tmp = np.linalg.norm(node_vec - com_vec_tmp)
                if node in com_dis:
                    if com_dis[node] > dis_tmp:
                        com_dis[node] = dis_tmp
                else:
                    com_dis[node] = dis_tmp

                # 存下综合最小欧氏距离delta变化最大的邻居节点
                if -delta_if_merge * (1 - beta) + com_dis[node] * beta < D:
                    D = -delta_if_merge * (1 - beta) + com_dis[node] * beta
                    delta = -delta_if_merge
                    com2 = [node]
                    vol3_after_merge = vol3
                    g3_after_merge = g3
                    neighbors_extra = neighbors_if_merge
            # print("dis:", dis)
        maxF1Res[-1].append(max_f1_precision)
        # print("F1-score:", F1_nodeNum_1[-1])
        msg = "beta: {}, Commnity count: {}, Max-F1: {}, max_f1_precision: {}, max_f1_recall_nodeNum: {}, embeddingFile: {}".format(beta, node_num, max_f1, max_f1_precision, max_f1_recall_nodeNum, vec_file_name[embedding])
        print(msg)
        logFile = "SEntropyParallel{}k{}.txt".format(graphFile,k)
        with open(logFile,'a') as f:
            f.write(msg + '\n')
        

if __name__ == "__main__":
    main()
        
