import networkx as nx
import numpy as np
import time
import random


# 利用SBM（随机块模型）生成无权有向联通图，划分为20个社区，
def sbm_construction():
    # 100个区块，每个区块的节点输为2690
    sizes = [2690 for i in range(20)]

    # 块之间连接边的概率
    no_diag_delta = 1e-6
    # 块内部连接边的概率
    diag_delta = 1e-3

    # 构建边的概率矩阵
    p = np.full(shape=(20, 20), fill_value=no_diag_delta)
    p[np.diag_indices_from(p)] = diag_delta

    G = nx.stochastic_block_model(sizes, p, seed=0, directed=True)
    print("Construction finished!")

    H = nx.DiGraph()
    H.add_nodes_from([str(i) for i in range(sum(sizes))])
    for u, v in G.edges:
        H.add_edge(str(u), str(v))

    print(H.number_of_nodes())
    print(H.number_of_edges())

    # nx.write_gml(H, "sbm_unweighted_20220715.gml")
    return H

# 给生成的有向图加入weight、datetime、feature等
# G = nx.read_gml("结构熵代码panner\sbm_unweighted_20220715.gml")
G = sbm_construction()
a = 0.52
size = G.number_of_edges() + 10  # 取比所需边数略大的size
lower = 300000  # 最小值
alpha = 0.1  # 与groundtruth之间的比例

# 使用帕累托模型生成与groundtruth权重分布类似的sbm权重（幂律分布）
# 符合2-8定律，The Pareto distribution must be greater than zero, and is unbounded above. It is also known as the "80-20 rule". In this distribution, 80 percent of the weights are in the lowest 20 percent of the range, while the other 20 percent fill the remaining 80 percent of the range.
weights = (np.random.pareto(a, size) + 1) * lower * alpha
weights = np.sort(weights)
weights = weights[:-10]
np.random.shuffle(weights)

# print(weights)

datetimes = []
# 随机生成datetime
s = (2008,4,8,0,0,0,0,0,0) #设置开始日期时间元组（2019-08-01 00：00：00）
e = (2016,4,28,23,59,59,0,0,0) #设置结束日期时间元组（2019-08-31 23：59：59)
start = time.mktime(s) #生成开始时间戳
end = time.mktime(e) #生成结束时间戳
for i in range(size):  
    t = random.randint(start, end) # 在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(t) # 将时间戳生成时间元组
    date = time.strftime("%Y%m%d%H%M%S", date_touple)
    # print(date)
    datetimes.append(date)

fvs= []
# 按属性随机生成feature
for i in range(size):  
    fs0vec = np.zeros(3)
    fs1vec = np.zeros(4)
    fs2vec = np.zeros(13)
    bankvec = np.zeros(676)
    moneyTypevec = np.zeros(2)
    cityvec = np.zeros(343)

    fs0vec[0] = 1
    fs1vec[0] = 1
    fs2vec[random.randint(0, fs2vec.shape[0]-1)] = 1
    bankvec[random.randint(0, bankvec.shape[0]-1)] = 1
    moneyTypevec[0] = 1
    cityvec[random.randint(0, cityvec.shape[0]-1)] = 1

    fv = np.concatenate((fs0vec,fs1vec, fs2vec, bankvec, moneyTypevec, cityvec), axis=None).tolist()
    fvs.append(fv)


i = 0
for u, v in G.edges:
    G.add_edge(u, v, weight=weights[i], datetime=str([datetimes[i]]), feature=str(fvs[i]))
    i += 1

print(G.number_of_nodes())
print(G.number_of_edges())

# nx.write_gml(G, "结构熵代码panner\sbm_weighted_20220715.gml")



#### combine_sbm_gt.ipynb
import networkx as nx
import random

# G1 = nx.read_gml("结构熵代码panner\sbm_weighted_20220715.gml")  # sbm图文件
G1 = G
G2 = nx.read_gml("结构熵代码panner\moneyLunderingDiGraphCombined.gml")  # gt文件


# preDatasetG = nx.read_gml("结构熵代码panner\combined_20220715.gml")

# print(G1.number_of_nodes(), G2.number_of_nodes())
# print(preDatasetG.number_of_nodes(), preDatasetG.number_of_nodes())

non_indegree = []
non_outdegree = []

for node in G2.nodes():
    if G2.in_degree(node) == 0:
        non_indegree.append(node)
    elif G2.out_degree(node) == 0:
        non_outdegree.append(node)

inner = non_indegree + non_outdegree
outer = [i for i in list(G2.nodes()) if i not in inner]
nonZeroIndegree = non_outdegree + outer

# print(len(non_indegree), len(non_outdegree), len(outer))
# 947  1316  427

map_inner = list(map(str, random.sample(range(G1.number_of_nodes()), len(inner))))
map_outer = list(map(str, (range(G1.number_of_nodes(), G1.number_of_nodes() + len(outer)))))
random.shuffle(map_outer)
# print(map_inner)
# print(map_outer)
# print(sorted(map_outer))

# 将入度或出度为0的点嵌入到sbm中（947+1316）
mapping = dict(zip(inner + outer, map_inner + map_outer))

# print(sorted(list(map(int, map_inner))))
gt = list(mapping.values())
gt_nonZeroIndegree = [mapping[node] for node in mapping if node in nonZeroIndegree]

G2 = nx.relabel_nodes(G2, mapping)
with open("结构熵代码panner\gt_20220729_lower{}.txt".format(lower),'w') as f:
    f.write(str(sorted(gt)))

# print(sorted(G2.nodes))
# print(sorted(G2.nodes, reverse=True))

# number_of_nodes = G1.number_of_nodes()
# for i in range(number_of_nodes, 60000):
#         G1.add_node(str(i), label = str(i))
       
for u, v in G2.edges:
    if G1.has_edge(u, v):
        G1[u][v]['weight'] = float(G1[u][v]['weight']) + float(G2[u][v]['weight'])
        G1[u][v]['datetime'] = str(eval(G1[u][v]['datetime']) + [G2[u][v]['datetime']])
        G1[u][v]['feature'] = str(list(np.array(eval(G1[u][v]["feature"])) + np.array(eval(G2[u][v]["feature"]))))
    else:
        # 保证graph中的id和label相同，不知道是否有用，但至少没坏处，不然分不清到底用的是哪个
        if(int(u) >= G1.number_of_nodes()):
            number_of_nodes = G1.number_of_nodes() 
            for i in range(number_of_nodes, int(u)+1):
                G1.add_node(str(i), label = str(i))
        if(int(v) >= G1.number_of_nodes()):
            number_of_nodes = G1.number_of_nodes() 
            for i in range(number_of_nodes, int(v)+1):
                G1.add_node(str(i), label = str(i))

        G1.add_edge(u, v, weight=G2[u][v]['weight'], datetime=G2[u][v]['datetime'], feature=G2[u][v]['feature'])

nx.write_gml(G1, "结构熵代码panner\combined_20220729_lower{}.gml".format(lower))