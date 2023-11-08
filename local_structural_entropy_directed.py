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
# from timebudget import timebudget
from multiprocessing import Pool
import multiprocessing
import argparse
import heapq
import time
import random

MININF = -99999999
# # 加权图
# G = nx.DiGraph()
# for i in range(8):
#     G.add_node(i, label = i)
# G.add_weighted_edges_from([(0,1,3.0), (0,2,4.0), (1,2,3.0), (1,3,8.0), (3,5,2.0), (4,5,4.0), (5,6,3.0), (4,7,5.0), (5,7,4.0), (3,6,7.5), (6,7,9.5)])

graphFile = "combined_20220729_lower300000.gml"
gtFile = "gt_20220729_lower300000.txt"

deltaOfNeighbor = []

G = nx.read_gml(graphFile)
with open(gtFile) as f:
    gt = eval(f.readline())

# choose start node candidate top 1000
# random.seed(42)
# startCandidate = random.sample(gt, 1000)
# # print(startCandidate)
# random.shuffle(startCandidate)
# print(startCandidate)
# exit()

# 读取embedding文件，并转换为numpy形式
def loadEmbeddingFile(embeddingFileName):
    embList = []
    with open(embeddingFileName, "r") as f:
        for i, line in enumerate(f.readlines()):
            nodeNum,vec = line.split(':')
            vec = vec.strip().split()
            embList.append(list(map(float, vec)))
            if(i != int(nodeNum)):
                print("Error extract embedding file: {} line {}, nodeNum: {}".format(embeddingFileName, i, nodeNum))
    embeddings = np.array(embList)
    return embeddings

# 计算社区S的邻居mu距离社区S中心点的距离
def neighbor2CommunityDistance(S, mu, embeddings):
    S = list(map(int, S))
    center = np.mean(embeddings[S], axis=0)
    distance = np.linalg.norm(center - embeddings[int(mu)])
    return distance


def Check(maxDelta, size):
    if(maxDelta < 1 or size > 4000):
        return -1
    return 1

# 统计图G的所有点的度数之和（indegree + outdegree）
def volOfGraph(G):
    degList = [deg for (node, deg) in G.in_degree()] + [deg for (node, deg) in G.out_degree()]
    volV = np.sum(np.array(degList))
    return volV

# 统计图G的特定节点的度数之和（indegree + outdegree）
def volOfGraphNodes(G, nodeList):
    degList = [deg for (node, deg) in G.in_degree(nbunch=nodeList)] + [deg for (node, deg) in G.out_degree(nbunch=nodeList)]
    volV = np.sum(np.array(degList))
    return volV

# 统计图G的由nodeList点集组成的诱导子图的度数之和（indegree + outdegree）
def volOfInducedSubGraph(G, nodeList):
    return volOfGraph(G.subgraph(nodeList))

# 统计图G中节点集合S和S其中一个邻居节点mu之间的weight之和
def weightBetweenSandMu(G, S, mu):
    neighborsOfMu = list(nx.all_neighbors(G, mu))
    neighborsOfMuInS = list(set(S) & set(neighborsOfMu))
    # 原始weight值的定义
    # weightList = [G[mu][node]['weight'] if G.has_edge(mu, node) else G[node][mu]['weight'] for node in neighborsOfMuInS]
    # 使用multigraph中给的平行边数量*weight，加大普通节点和洗钱节点之间的差距
    weightList = [G[mu][node]['weight'] * len(eval(G[mu][node]['datetime'])) if G.has_edge(mu, node) else G[node][mu]['weight']* len(eval(G[node][mu]['datetime'])) for node in neighborsOfMuInS]
    weight = np.sum(np.array(weightList))
    return weight

# 修剪neighbor节点，加快运行速度
def trimNeighbor(neighborCandidates):
    realCandidates = sorted(neighborCandidates.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:]
    return [node for (node, weight) in realCandidates]

def maxNIndex(deltaOfNeighbor, N):
    if(N > len(deltaOfNeighbor)):
        N = len(deltaOfNeighbor)
    max_number = heapq.nlargest(N, deltaOfNeighbor) 
    max_index = []
    for t in max_number:
        index = deltaOfNeighbor.index(t)
        max_index.append(index)
        deltaOfNeighbor[index] = MININF
    return max_index

# 评估结果，计算F1-score, recall_nodeNum, precision_nodeNum三个指标
def evaluate(gt, seq):
    # calculate F-1 Score:
    TruePositive = len(set(gt) & set(seq))
    FalsePositive = len(set(seq) - set(gt))
    FalseNegative = len(set(gt) - set(seq))
    f1_score = 2 * TruePositive / (2 * TruePositive + FalsePositive + FalseNegative)
    # print(num_seq, num_gt, num_right)

    # calculate recall_nodeNum:
    num_seq = len(seq)
    num_gt = len(gt)
    num_right = len(set(gt) & set(seq))

    if num_gt != 0:
        recall_nodeNum = num_right / num_gt
    else:
        recall_nodeNum = 0

    # calculate precision_nodeNum:
    if num_seq != 0:
        precision_nodeNum = num_right / num_seq
    else:
        precision_nodeNum = 0
    
    return f1_score, recall_nodeNum, precision_nodeNum

def SEntropy(G):
    volV = volOfGraph(G)
    S = ['8608']
    NeighborsOfS = []
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(Check()):        
        deltaOfNeighbor = [MININF for i in range(len(NeighborsOfS))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        for i,mu in enumerate(NeighborsOfS):
            SandMu = S + [mu]
            delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
                    -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            deltaOfNeighbor[i] = delta
        
        MaxIndex = deltaOfNeighbor.index(max(deltaOfNeighbor))
        muCandidate = NeighborsOfS[MaxIndex]
        S.append(muCandidate)
        muNeighbor = list(nx.all_neighbors(G, muCandidate))
        # neighborCandidate = list(set(muNeighbor) - set(S))
        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))
        print("Community size: {}, neighbor size: {}".format(len(S), len(NeighborsOfS)))
        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        print("f1: {}, recall: {}, precision: {}".format(f1_score, recall_nodeNum, precision_nodeNum))
        if(len(S) >= G.number_of_nodes() or len(NeighborsOfS) == 0):
            break

# 仅仅依靠degree*weight来筛选节点会导致筛选错误，F1下降
def SEntropyTrim(G):
    volV = volOfGraph(G)
    S = ['8608']
    maxDelta = 0
    # 连续加入10个节点都不符合check条件，循环结束
    breakAlarm = 10
    NeighborsOfS = []
    maxF1 = 0
    # 保存neighbor的weight*degree 的值，作为剪枝依据，取值最大的前N个
    neighborCandidates = {}
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
        for u in list(nx.all_neighbors(G, node)):
            neighborCandidates[u] = (G[node][u]['weight'] * G.degree[u] if G.has_edge(node, u) else G[u][node]['weight']) * G.degree[u]

    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(breakAlarm): 
        candidates  = trimNeighbor(neighborCandidates)    
        deltaOfNeighbor = [MININF for i in range(len(candidates))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        
        for i,mu in enumerate(candidates):
            SandMu = S + [mu]
            delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
                    -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            deltaOfNeighbor[i] = delta
        print(min(deltaOfNeighbor), max(deltaOfNeighbor))
        maxDelta = max(deltaOfNeighbor)

        breakAlarm = breakAlarm + Check(maxDelta, len(S))
        breakAlarm = 10 if breakAlarm > 10 else breakAlarm

        MaxIndex = deltaOfNeighbor.index(max(deltaOfNeighbor))
        muCandidate = candidates[MaxIndex]
        S.append(muCandidate)
        muNeighbor = list(nx.all_neighbors(G, muCandidate))
        # neighborCandidate = list(set(muNeighbor) - set(S))
        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))

        neighborCandidates.pop(muCandidate) 
        for u in list(nx.all_neighbors(G, muCandidate)):
            if(u not in S):
                if(u not in neighborCandidates.keys()):
                    neighborCandidates[u] = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                else:
                    val = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                    neighborCandidates[u] = neighborCandidates[u] if(neighborCandidates[u] > val) else val

        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        maxF1 = f1_score if f1_score > maxF1 else maxF1
        print("Community size: {}, neighbor size: {}, candidates size: {}, f1: {}, recall: {}, precision: {}, MaxF1: {}".format(len(S), len(NeighborsOfS), len(neighborCandidates.keys()), f1_score, recall_nodeNum, precision_nodeNum, maxF1))
        
        if(len(S) >= G.number_of_nodes() or len(NeighborsOfS) == 0):
            break

def SEntropyTrimEmbedding(G, embeddings, embeddingFileName):
    volV = volOfGraph(G)
    S = ['8608']
    maxDelta = 0
    # 连续加入10个节点都不符合check条件，循环结束
    breakAlarm = 10
    NeighborsOfS = []
    maxF1 = 0
    maxF1_recall = 0
    maxF1_precision = 0
    maxF1_communitySize = 0
    # 保存neighbor的weight*degree 的值，作为剪枝依据，取值最大的前N个
    neighborCandidates = {}
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
        for u in list(nx.all_neighbors(G, node)):
            neighborCandidates[u] = (G[node][u]['weight'] * G.degree[u] if G.has_edge(node, u) else G[u][node]['weight']) * G.degree[u]

    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(breakAlarm): 
        candidates  = trimNeighbor(neighborCandidates)   
        deltaOfNeighbor = [MININF for i in range(len(candidates))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        
        for i,mu in enumerate(candidates):
            SandMu = S + [mu]
            delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
                    -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            embDistance = neighbor2CommunityDistance(S, mu, embeddings)
            deltaOfNeighbor[i] = delta / embDistance
        print(min(deltaOfNeighbor), max(deltaOfNeighbor))
        maxDelta = max(deltaOfNeighbor)

        breakAlarm = breakAlarm + Check(maxDelta, len(S))
        breakAlarm = 10 if breakAlarm > 10 else breakAlarm

        MaxIndex = deltaOfNeighbor.index(max(deltaOfNeighbor))
        muCandidate = candidates[MaxIndex]
        S.append(muCandidate)

        muNeighbor = list(nx.all_neighbors(G, muCandidate))
        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))

        neighborCandidates.pop(muCandidate) 
        for u in list(nx.all_neighbors(G, muCandidate)):
            if(u not in S):
                if(u not in neighborCandidates.keys()):
                    neighborCandidates[u] = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                else:
                    val = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                    neighborCandidates[u] = neighborCandidates[u] if(neighborCandidates[u] > val) else val

        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        if (f1_score > maxF1) :
            maxF1 = f1_score
            maxF1_recall = recall_nodeNum
            maxF1_precision = precision_nodeNum
            maxF1_communitySize = len(S)
        print("Community size: {}, neighbor size: {}, candidates size: {}, f1: {}, recall: {}, precision: {}, MaxF1: {}".format(len(S), len(NeighborsOfS), len(neighborCandidates.keys()), f1_score, recall_nodeNum, precision_nodeNum, maxF1))
        
        if(len(S) >= G.number_of_nodes() or len(NeighborsOfS) == 0):
            break
    with open("local_structural_entropy.log", "a") as f:
        msg = "{},{},{},{},{}\n".format(embeddingFileName, maxF1, maxF1_recall, maxF1_precision, maxF1_communitySize)
        f.write(msg)

def SEntropyTrimEmbeddingMaxN(G, embeddings, embeddingFileName, beta):
    tStart = time.time()
    volV = volOfGraph(G)
    S = ['8608']
    maxDelta = 0
    # 连续加入breakAlarm个节点都不符合check条件，循环结束
    breakAlarm = 5000
    NeighborsOfS = []
    maxF1 = 0
    maxF1_recall = 0
    maxF1_precision = 0
    maxF1_communitySize = 0
    # 保存neighbor的weight*degree 的值，作为剪枝依据，取值最大的前N个
    neighborCandidates = {}
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
        for u in list(nx.all_neighbors(G, node)):
            neighborCandidates[u] = (G[node][u]['weight'] * G.degree[u] if G.has_edge(node, u) else G[u][node]['weight']) * G.degree[u]

    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(breakAlarm): 
        # 放弃使用剪枝策略
        # candidates  = trimNeighbor(neighborCandidates)   
        candidates = list(neighborCandidates.keys())
        deltaOfNeighbor = [MININF for i in range(len(candidates))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        
        for i,mu in enumerate(candidates):
            SandMu = S + [mu]
            # use both delta and distance information
            delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
                    -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            embDistance = neighbor2CommunityDistance(S, mu, embeddings)
            deltaOfNeighbor[i] = 44539960900*delta / embDistance

            # # use only delta entropy information
            # delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
            #         -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            # deltaOfNeighbor[i] = delta

            # # use only distance information
            # embDistance = neighbor2CommunityDistance(S, mu, embeddings)
            # deltaOfNeighbor[i] = 44539960900 / embDistance
        # print("deltaOfNeighbor min:{}, max:{}, distance:{}".format(min(deltaOfNeighbor), max(deltaOfNeighbor), embDistance))
        # maxDelta = max(deltaOfNeighbor)

        # breakAlarm = breakAlarm + Check(maxDelta, len(S))
        # breakAlarm = 5 if breakAlarm > 5 else breakAlarm

        # 加入最大的前N%个候选者
        MaxIndex = maxNIndex(deltaOfNeighbor, math.ceil(len(deltaOfNeighbor)*beta))
        # MaxIndex = maxNIndex(deltaOfNeighbor, 1)
        muCandidates = [candidates[i] for i in MaxIndex]
        S.extend(muCandidates)

        muNeighbor = []
        for node in muCandidates:
            muNeighbor.extend(list(nx.all_neighbors(G, node)))

        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))

        for node in muCandidates:
            neighborCandidates.pop(node) 
        for muCandidate in muCandidates:
            for u in list(nx.all_neighbors(G, muCandidate)):
                if(u not in S):
                    if(u not in neighborCandidates.keys()):
                        neighborCandidates[u] = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                    else:
                        val = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                        neighborCandidates[u] = neighborCandidates[u] if(neighborCandidates[u] > val) else val

        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        if (f1_score > maxF1) :
            maxF1 = f1_score
            maxF1_recall = recall_nodeNum
            maxF1_precision = precision_nodeNum
            maxF1_communitySize = len(S)
        print("Beta: {}, Community size: {}, neighbor size: {}, candidates size: {}, f1: {}, recall: {}, precision: {}, MaxF1: {}, Running Time: {}".format(beta, len(S), len(NeighborsOfS), len(candidates), f1_score, recall_nodeNum, precision_nodeNum, maxF1, time.time()-tStart))

        # with open(embeddingFileName + "_SEntropy_Embedding.res", 'w') as f:
        #     f.write(str(S))

        # with open("combined_20220729_lower300000.gml_SEntropy.res", 'w') as f:
        #     f.write(str(S))

        with open("local_structural_entropy_onlyentropyBetaRatioTest.csv", "a") as f:
            # msg = "{},{},{},{},{},{}\n".format(embeddingFileName, maxF1, maxF1_recall, maxF1_precision, maxF1_communitySize, time.time()-tStart)
            msg = "{},{},{},{},{},{},{},{},{},{}\n".format(embeddingFileName, beta, maxF1, maxF1_recall, maxF1_precision, maxF1_communitySize, f1_score, recall_nodeNum, precision_nodeNum, time.time()-tStart)
            f.write(msg)
        
        if(len(S) >= 4000 or len(NeighborsOfS) == 0):
            print("NeighborsOfS size: {}, exit".format(len(NeighborsOfS)))
            break
        if(maxF1/f1_score > 1.02):
            print("f1 decreased! exit!")
            break

def SEntropyTrimEmbeddingMaxNStartFromNNodes(G, embeddings, embeddingFileName, beta, startCommSize):
    tStart = time.time()
    volV = volOfGraph(G)
    with open('startNodeCandidateTop1000.txt','r') as f:
        sCandidates = eval(f.readline())

    S = sCandidates[:startCommSize]
    maxDelta = 0
    # 连续加入breakAlarm个节点都不符合check条件，循环结束
    breakAlarm = 5000
    NeighborsOfS = []
    maxF1 = 0
    maxF1_recall = 0
    maxF1_precision = 0
    maxF1_communitySize = 0
    # 保存neighbor的weight*degree 的值，作为剪枝依据，取值最大的前N个
    neighborCandidates = {}
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
        for u in list(nx.all_neighbors(G, node)):
            neighborCandidates[u] = (G[node][u]['weight'] * G.degree[u] if G.has_edge(node, u) else G[u][node]['weight']) * G.degree[u]

    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(breakAlarm): 
        # 放弃使用剪枝策略
        # candidates  = trimNeighbor(neighborCandidates)   
        candidates = list(neighborCandidates.keys())
        deltaOfNeighbor = [MININF for i in range(len(candidates))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        
        for i,mu in enumerate(candidates):
            SandMu = S + [mu]
            # use both delta and distance information
            delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
                    -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            embDistance = neighbor2CommunityDistance(S, mu, embeddings)
            deltaOfNeighbor[i] = 44539960900*delta / embDistance

            # # use only delta entropy information
            # delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
            #         -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))
            # deltaOfNeighbor[i] = delta

            # # use only distance information
            # embDistance = neighbor2CommunityDistance(S, mu, embeddings)
            # deltaOfNeighbor[i] = 44539960900 / embDistance
        # print("deltaOfNeighbor min:{}, max:{}, distance:{}".format(min(deltaOfNeighbor), max(deltaOfNeighbor), embDistance))
        # maxDelta = max(deltaOfNeighbor)

        # breakAlarm = breakAlarm + Check(maxDelta, len(S))
        # breakAlarm = 5 if breakAlarm > 5 else breakAlarm

        # 加入最大的前N%个候选者
        MaxIndex = maxNIndex(deltaOfNeighbor, math.ceil(len(deltaOfNeighbor)*beta))
        # MaxIndex = maxNIndex(deltaOfNeighbor, 1)
        muCandidates = [candidates[i] for i in MaxIndex]
        S.extend(muCandidates)

        muNeighbor = []
        for node in muCandidates:
            muNeighbor.extend(list(nx.all_neighbors(G, node)))

        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))

        for node in muCandidates:
            neighborCandidates.pop(node) 
        for muCandidate in muCandidates:
            for u in list(nx.all_neighbors(G, muCandidate)):
                if(u not in S):
                    if(u not in neighborCandidates.keys()):
                        neighborCandidates[u] = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                    else:
                        val = (G[muCandidate][u]['weight']* G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                        neighborCandidates[u] = neighborCandidates[u] if(neighborCandidates[u] > val) else val

        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        if (f1_score > maxF1) :
            maxF1 = f1_score
            maxF1_recall = recall_nodeNum
            maxF1_precision = precision_nodeNum
            maxF1_communitySize = len(S)
        print("Beta: {}, StartCommunitySize:{}, Community size: {}, neighbor size: {}, candidates size: {}, f1: {}, recall: {}, precision: {}, MaxF1: {}, Running Time: {}".format(beta, startCommSize, len(S), len(NeighborsOfS), len(candidates), f1_score, recall_nodeNum, precision_nodeNum, maxF1, time.time()-tStart))

        # with open(embeddingFileName + "_SEntropy_Embedding.res", 'w') as f:
        #     f.write(str(S))

        # with open("combined_20220729_lower300000.gml_SEntropy.res", 'w') as f:
        #     f.write(str(S))

        with open("local_structural_entropy_entropyEmbedding_communitySizeTest.csv", "a") as f:
            # msg = "{},{},{},{},{},{}\n".format(embeddingFileName, maxF1, maxF1_recall, maxF1_precision, maxF1_communitySize, time.time()-tStart)
            msg = "{},{},{},{},{},{},{},{},{},{},{}\n".format(embeddingFileName, beta, startCommSize, maxF1, maxF1_recall, maxF1_precision, maxF1_communitySize, f1_score, recall_nodeNum, precision_nodeNum, time.time()-tStart)
            f.write(msg)
        
        if(len(S) >= 4000 or len(NeighborsOfS) == 0):
            print("NeighborsOfS size: {}, exit".format(len(NeighborsOfS)))
            break
        if(maxF1/f1_score > 1.02):
            print("f1 decreased! exit!")
            break

def deltaEntropy(G, S, mu, volV, deltaOfNeighbor, i, share_lock):    
    # G, S, mu, volV, deltaOfNeighbor, i = paras[0],paras[1],paras[2],paras[3],paras[4],paras[5]
    # print("process id: {}".format(i))
    SandMu = S + [mu]
    delta = -1/volV *(volOfInducedSubGraph(G, SandMu) * math.log(volOfGraphNodes(G, SandMu),2)
            -2*weightBetweenSandMu(G, S, mu) - volOfInducedSubGraph(G, S)* math.log(volOfGraphNodes(G, S),2))

    # 并发计算完成之后，所有进程顺序访问deltaOfNeighbor数组，将计算结果写入数组
    # 这一步虽然为串行操作，但计算耗时主要集中在上步的delta运算上，所以并不会减缓并行效率
    share_lock.acquire()
    deltaOfNeighbor[i] = delta
    share_lock.release()
    # print(deltaOfNeighbor)

# 在剪枝的基础上对计算S的邻居节点加入前后的delta值进行并行化处理
# 原本以为可以加速，但实际把整体运算速度急剧拖慢
# 原因猜测是初始化multiprocessing时耗时过多，比计算单个delta要长很多，导致并行反而速度变慢
def SEntropyTrimParallel(G):
    volV = volOfGraph(G)
    S = ['8608']
    # S = [0]
    maxDelta = 0
    # 连续加入10个节点都不符合check条件，循环结束
    breakAlarm = 10
    NeighborsOfS = []
    maxF1 = 0
    # 保存neighbor的weight*degree 的值，作为剪枝依据，取值最大的前N个
    neighborCandidates = {}
    for node in S:
        NeighborsOfS.extend(list(nx.all_neighbors(G, node)))
        for u in list(nx.all_neighbors(G, node)):
            neighborCandidates[u] = (G[node][u]['weight'] * G.degree[u] if G.has_edge(node, u) else G[u][node]['weight']) * G.degree[u]

    NeighborsOfS = list(set(NeighborsOfS) - set(S))
    while(breakAlarm): 
        candidates  = trimNeighbor(neighborCandidates)    
        deltaOfNeighbor = multiprocessing.Manager().list([MININF for i in range(len(candidates))])
        # deltaOfNeighbor = [MININF for i in range(len(candidates))]
        # print("NeighborsOfS: {}".format(NeighborsOfS))
        
        # 使用multiprocessing进行并行计算加速处理速度
        paraList = []
        for i,mu in enumerate(candidates):
            paraList.append((G, S, mu, volV, deltaOfNeighbor, i))

        share_lock = multiprocessing.Manager().Lock()
        process_list = []

        for i, paras in enumerate(paraList):
            tmp_process = multiprocessing.Process(target=deltaEntropy,args=(G, S, mu, volV, deltaOfNeighbor, i, share_lock))
            process_list.append(tmp_process)

        for process in process_list:
            process.start()
        for process in process_list:
            process.join()

        
        # print(min(deltaOfNeighbor), max(deltaOfNeighbor))
        maxDelta = max(deltaOfNeighbor)

        breakAlarm = breakAlarm + Check(maxDelta, len(S))
        breakAlarm = 10 if breakAlarm > 10 else breakAlarm

        MaxIndex = deltaOfNeighbor.index(max(deltaOfNeighbor))
        muCandidate = candidates[MaxIndex]
        S.append(muCandidate)
        muNeighbor = list(nx.all_neighbors(G, muCandidate))
        # neighborCandidate = list(set(muNeighbor) - set(S))
        NeighborsOfS = set(NeighborsOfS).union(set(muNeighbor))
        NeighborsOfS = list(NeighborsOfS.difference(set(S)))

        neighborCandidates.pop(muCandidate) 
        for u in list(nx.all_neighbors(G, muCandidate)):
            if(u not in S):
                if(u not in neighborCandidates.keys()):
                    neighborCandidates[u] = (G[muCandidate][u]['weight'] * G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                else:
                    val = (G[muCandidate][u]['weight'] * G.degree[u] if G.has_edge(muCandidate, u) else G[u][muCandidate]['weight']) * G.degree[u]
                    neighborCandidates[u] = neighborCandidates[u] if(neighborCandidates[u] > val) else val

        f1_score, recall_nodeNum, precision_nodeNum = evaluate(gt, S)
        maxF1 = f1_score if f1_score > maxF1 else maxF1
        print("Community size: {}, neighbor size: {}, candidates size: {}, f1: {}, recall: {}, precision: {}, MaxF1: {}".format(len(S), len(NeighborsOfS), len(neighborCandidates.keys()), f1_score, recall_nodeNum, precision_nodeNum, maxF1))
        
        if(len(S) >= G.number_of_nodes() or len(NeighborsOfS) == 0):
            break

def main():
    embFiles = ["combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_64.emb"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.01)
    # 44 embedding files in total, so the embeddingPtr should in [0,43]
    parser.add_argument("--embeddingPtr", type=int, default=2)
    parser.add_argument("--maxCommunitySize", type=int, default=3000)
    parser.add_argument("--startCommSize", type=int, default=2)
    args = parser.parse_args()
    # SEntropyTrim(G)
    embeddingFileName = embFiles[args.embeddingPtr]
    embeddingFileName = "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_32.emb"
    # if('dgi' in embeddingFileName or 'mvgrl' in embeddingFileName):
    embeddings = loadEmbeddingFile(embeddingFileName)
    SEntropyTrimEmbeddingMaxNStartFromNNodes(G, embeddings, embeddingFileName, args.beta, args.startCommSize)


if __name__ == "__main__":
    main()