import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

GCombined = nx.read_gml("结构熵代码panner\moneyLunderingDiGraphCombined.gml")
GRaw = nx.read_gml("结构熵代码panner\moneyLunderingGraphGroundTruth.gml")
def diGraphWeightAnalysis(G):
    weightList = []
    for (u,v) in G.edges():
        weightList.append(G[u][v]['weight'])
    weightList.sort(reverse=True)
    weightArray = np.array(weightList)
    # weightArray = weightArray.sort(reverse=True)
    value = "Len:{}, Max: {}, Min:{}, \nMean:{}, Media:{}".format(weightArray.shape, weightArray.max(),weightArray.min(),float(weightArray.mean()),np.median(weightArray))
    print(value)
    
    delta = 500000
    bins = [i*delta for i in range(int(int(weightArray.max())/delta + 2))]
    weightDist = np.zeros(len(bins))
    for item in weightArray.tolist():
        weightDist[int(item/delta)] += 1
    
    plt.clf()
    plt.xlabel('EdgeID')
    plt.ylabel('Weight')
    plt.title('Weight on GroundTruth DiGraph Combined\n'+value)
    plt.grid(True)
    plt.loglog(int(weightArray.max()), int(weightDist.max()),color='r',linewidth=1)
    plt.scatter(bins, weightDist.tolist())
    plt.savefig("WeightOnGTDiGraphCombined.jpg", format='jpg', dpi=1000)

def multiGraphWeightAnalysis(G):
    weightList = []
    for u, v, key,data in G.edges(data=True, keys=True):
        weightList.append(data["weight"])
    weightList.sort(reverse=True)
    weightArray = np.array(weightList)
    # weightArray = weightArray.sort(reverse=True)
    value = "Len:{}, Max: {}, Min:{}, \nMean:{}, Media:{}".format(weightArray.shape, weightArray.max(),weightArray.min(),float(weightArray.mean()),np.median(weightArray))
    print(value)
    
    delta = 100000
    bins = [i*delta for i in range(int(int(weightArray.max())/delta + 2))]
    weightDist = np.zeros(len(bins))
    for item in weightArray.tolist():
        weightDist[int(item/delta)] += 1
    
    plt.clf()
    plt.xlabel('Weight')
    plt.ylabel('Edge Count')
    plt.title('Weight on GroundTruth MultiGraph RAW\n'+value)
    plt.grid(True)
    plt.loglog(int(weightArray.max()), int(weightDist.max()),color='r',linewidth=1)
    plt.scatter(bins, weightDist.tolist())
    # plt.gca().invert_xaxis()#x轴反转，大的值在前面，小的值在后面
    plt.savefig("WeightOnGTRAW.jpg", format='jpg', dpi=200)
    
def ParetoWeightAnalysis(G):
    
    # 给生成的有向图加入weight、datetime、feature等
    # G = nx.read_gml("结构熵代码panner\sbm_unweighted_20220715.gml")
    # G = sbm_construction()
    a = 0.52
    size = G.number_of_edges() + 10  # 取比所需边数略大的size
    lower = 3  # 最小值
    alpha = 0.1  # 与groundtruth之间的比例

    # 使用帕累托模型生成与groundtruth权重分布类似的sbm权重（幂律分布）
    # 符合2-8定律，The Pareto distribution must be greater than zero, and is unbounded above. It is also known as the "80-20 rule". In this distribution, 80 percent of the weights are in the lowest 20 percent of the range, while the other 20 percent fill the remaining 80 percent of the range.
    weights = (np.random.pareto(a, size) + 1) * lower * alpha
    weights = np.sort(weights,axis= 0)
    weights = weights[:-10][::-1]
    # np.random.shuffle(weights)
    weightArray = weights

    value = "Len:{}, Max: {}, Min:{}, \nMean:{}, Media:{}".format(weightArray.shape, weightArray.max(),weightArray.min(),float(weightArray.mean()),np.median(weightArray))
    print(value)
    
    delta = 500
    bins = [i*delta for i in range(int(int(weightArray.max())/delta + 2))]
    weightDist = np.zeros(len(bins))
    for item in weightArray.tolist():
        weightDist[int(item/delta)] += 1
    
    plt.clf()
    plt.xlabel('EdgeID')
    plt.ylabel('Weight')
    plt.title('Weight on Pareto Generator a=0.52 lower=3\n'+value)
    # plt.bar(range(len(weightArray)), weightArray)
    plt.grid(True)
    plt.loglog(int(weightArray.max()), int(weightDist.max()),color='r',linewidth=1)
    plt.scatter(bins, weightDist.tolist())
    plt.savefig("ParetoValues.jpg", format='jpg', dpi=1000)

def featureVisualize(G):
    fs0vec = np.zeros(3)
    fs1vec = np.zeros(4)
    fs2vec = np.zeros(13)
    bankvec = np.zeros(676)
    moneyTypevec = np.zeros(2)
    cityvec = np.zeros(343)

    for u, v, in G.edges:
        fv = eval(G[u][v]['feature'])
        fs0vec += np.array(fv[:3])
        fs1vec += np.array(fv[3:7])
        fs2vec += np.array(fv[7:20])
        bankvec += np.array(fv[20:696])
        moneyTypevec += np.array(fv[696:698])
        cityvec += np.array(fv[698:1041])
        
    fs0vec /= np.sum(fs0vec)
    fs1vec /= np.sum(fs1vec)
    fs2vec /= np.sum(fs2vec)
    bankvec /= np.sum(bankvec)
    moneyTypevec /= np.sum(moneyTypevec)
    cityvec /= np.sum(cityvec)

    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('JIAOYIFANGSHI vec0')
    plt.bar(range(len(fs0vec)), fs0vec)
    plt.savefig("交易方式Vec0.jpg")

    plt.clf()
    plt.title('JIAOYIFANGSHI vec1')
    plt.bar(range(len(fs1vec)), fs1vec)
    plt.savefig("交易方式Vec1.jpg")

    plt.clf()
    plt.title('JIAOYIFANGSHI vec2')
    plt.bar(range(len(fs2vec)), fs2vec)
    plt.savefig("交易方式Vec2.jpg")

    plt.clf()
    plt.title('Bank vec0')
    plt.bar(range(len(bankvec)), bankvec)
    plt.savefig("银行vec.jpg")

    plt.clf()
    plt.title('MoneyType vec')
    plt.bar(range(len(moneyTypevec)), moneyTypevec)
    plt.savefig("币种.jpg")

    plt.clf()
    plt.title('City Vec')
    plt.bar(range(len(cityvec)), cityvec)
    plt.savefig("城市Vec.jpg")

    print(fs0vec)
    print(fs1vec)
    print(fs2vec)
    print(bankvec)
    print(moneyTypevec)
    print(cityvec)
    
def main():
    ParetoWeightAnalysis(GCombined)
    diGraphWeightAnalysis(GCombined)
    multiGraphWeightAnalysis(GRaw)
    
if __name__ == "__main__":
    main()
