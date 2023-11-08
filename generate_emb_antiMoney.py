import numpy as np
from cogdl import pipeline
import networkx as nx
import os
import threading
import time
import math

# from classify import read_node_label, Classifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import SpectralClustering

# os.environ["CUDA_VISIBLE_DEVICES"]="5,6" ###指定此处为-1即可
# CUDA_VISIBLE_DEVICES=5 python generate_emb_antiMoney.py  使用这个有用

# build a pipeline for generating embeddings
# pass model name with its hyper-parameters to this API

modelNames = ['prone','hope','netsmf','netmf','line','deepwalk', 'node2vec', "mvgrl", "dgi"]
# modelNames = ['graph2vec','complex','dgk','distmult','dngr','gatne','transe']

# modelNames = ["mvgrl", "dgi", "grace", "unsup_graphsage"]


dims = [32,64,128,256, 512]
# dims = [512]
numberFetures = [-1]

#these models can not run, exist some bugs
# , 'sdne', 'grarep', 'dngr'
# ['gcc','sdne', 'grarep', 'dngr']

gmlFileName = 'combined_20220729_lower300000.gml'
# gmlModelName = gmlFileName + '.model'
# gmlResName = gmlFileName+'.res.txt'
communityNum = 30

# this method is too slow
def getEdgeNodeList():
    G = nx.read_gml(gmlFileName,destringizer=float)
    nodeList = []
    for u,v in G.edges():
        if u not in nodeList:
            nodeList.append(u)
        if v not in nodeList:
            nodeList.append(v)
    nodeList.sort()
    print("Number of edged nodes: ", len(nodeList))
    return nodeList

def getEdgeIndex():
    if(not os.path.exists(gmlFileName+".npz")):
        G = nx.read_gml(gmlFileName,destringizer=float)
        # G = nx.read_gml(gmlFileName)
        print("Number of nodes: ", G.number_of_nodes())
        print("Number of edges: ", G.number_of_edges())
        edge_index = np.array([[int(u),int(v)] for (u, v) in G.edges()])
        edge_weight = np.array([float(G[u][v]['weight']) for (u, v) in G.edges()])

        nodeList = []
        # for item in G.degree():
            # if(item[1] != 0):
            #     nodeList.append(int(item[0]))
        

        # construct features
        featureList = []
        for node in G.nodes():
            nodeList.append(node)
            totalWeight = 0
            nodeFeature = np.zeros(1041)
            # for neig in G[node]:
            #     totalWeight += G[node][neig]['weight']
            for neig in G[node]:
                nodeFeature += G[node][neig]['weight'] * np.array(eval(G[node][neig]["feature"]))
                # nodeFeature += G[node][neig]['weight']
            featureList.append(nodeFeature)
        # print("Number of nodes on edges: ", len(nodeList))

        feaNumpy = np.stack(featureList)
        nodeFeatures = feaNumpy[:, [not np.all(feaNumpy[:, i] == 0) for i in range(feaNumpy.shape[1])]]
        print(nodeFeatures)
        print("nodeFeatures shape: ", nodeFeatures.shape)

        # features = torch.tensor(feaNumpy)
        # features = features.float()
        np.savez(gmlFileName+".npz",edge_index=edge_index, edge_weight=edge_weight, nodeList=nodeList, nodeFeatures=nodeFeatures)
    else:
        npzfile=np.load(gmlFileName+".npz")
        edge_index = npzfile['edge_index']
        edge_weight = npzfile['edge_weight']
        nodeList = npzfile['nodeList']
        nodeFeatures = npzfile['nodeFeatures']
        print("edge_index shape:", edge_index.shape)
        print("edge_weight shape:", edge_weight.shape)
        print("nodeList shape:", nodeList.shape)
        print("nodeFeatures shape:", nodeFeatures.shape)

    return edge_index, edge_weight, nodeList, nodeFeatures


def singleModel(edge_index, edge_weight, nodeList, nodeFeatures, md, numFeatures, hiddenSize):
    print("=========================================================================")
    print(time.strftime("%D %H:%M:%S", time.localtime()) + " Model Name : "+ md + "_feature_" + str(numFeatures) + "_dim_" +str(hiddenSize))
    tStart = time.time()

    embeddingFileName = gmlFileName + "_" + md + "_feature_" + str(numFeatures) + "_dim_" +str(hiddenSize)+ ".emb"
    

    if(not os.path.exists(embeddingFileName)):
        print("Generate embeddings")
        generator = pipeline("generate-emb", model=md, no_test=True,num_features=numFeatures, hidden_size=hiddenSize)
        # embeddings = generator(edge_index, edge_weight=edge_weight, x=nodeFeatures)
        if(md in ["dgi", "mvgrl", "grace", "unsup_graphsage"]):
            embeddings = generator(edge_index, x=nodeFeatures)
            # embeddings = generator(edge_index, edge_weight=edge_weight,  x=nodeFeatures)
        else:
            embeddings = generator(edge_index, edge_weight=edge_weight)

        print("embedding shape: ",embeddings.shape)
        
        tEmbedding = time.time()
        print(time.strftime("%D %H:%M:%S", time.localtime()) +" Done model embedding {}, time used: {}s\n".format(md, tEmbedding-tStart))
        with open("log.txt", "a") as f:
            f.write("{} {} embedding time: {}\n".format(time.strftime("%D %H:%M:%S", time.localtime()),embeddingFileName, tEmbedding-tStart))

        print("Write embeddings")
        with open(embeddingFileName, "w") as f:
            for i, vec in enumerate(embeddings):
                vecStr = " ".join([str(i) for i in vec])
                f.write("{}:{}\n".format(int(nodeList[i]), vecStr))
                # f.write("{}:{}\n".format(i, vecStr))
    else:
        print("Embedding file exists, use old embedding file!")
        embList = []
        with open(embeddingFileName, "r") as f:
            for line in f.readlines():
                line = line.split(':')[1].strip()
                line = line.split()
                embList.append(list(map(float, line)))
        embeddings = np.array(embList)

    # embeddings = outputs
    print("K-Means Classification")
    kmeans = KMeans(n_clusters=communityNum, random_state=0).fit(embeddings)
    print("Write classified result")
    resFileName = gmlFileName + "_" + md + "_kmeans" + "_feature_" + str(numFeatures) + "_dim_" +str(hiddenSize)+ ".res"
    with open(resFileName, "w") as f:
        for i, label in enumerate(kmeans.labels_):
            # print(i,label)
            f.write("{} {}\n".format(int(nodeList[i]), label))
            # f.write("{} {}\n".format(i, label))
    tEnd = time.time()
    print(time.strftime("%D %H:%M:%S", time.localtime()) +" Done model {}, time used: {}s\n".format(md, tEnd-tStart))
    with open("log.txt", "a") as f:
        f.write("{} {} {}\n".format(time.strftime("%D %H:%M:%S", time.localtime()),resFileName, tEnd-tStart))
    print("=========================================================================")

    # # embeddings = outputs
    # resFileName = gmlFileName + "_" + md + "_spectral" + "_feature_" + str(numFeatures) + "_dim_" +str(hiddenSize)+ ".res"
    # if(not os.path.exists(resFileName)):
    #     print("Spectral Classification")
    #     kmeans = SpectralClustering(n_clusters=communityNum, random_state=0).fit(embeddings)
    #     print("Write classified result")
    #     # resFileName = gmlFileName + "_" + md + "_spectral" + "_feature_" + str(numFeatures) + "_dim_" +str(hiddenSize)+ ".res"
    #     with open(resFileName, "w") as f:
    #         for i, label in enumerate(kmeans.labels_):
    #             # print(i,label)
    #             f.write("{} {}\n".format(int(nodeList[i]), label))
    #     tEnd = time.time()
    #     print(time.strftime("%D %H:%M:%S", time.localtime()) +" Done model {}, time used: {}s\n".format(md, tEnd-tStart))
    #     with open("log.txt", "a") as f:
    #         f.write("{} {} {}\n".format(time.strftime("%D %H:%M:%S", time.localtime()),resFileName, tEnd-tStart))
    #     print("=========================================================================")
    

def main():
    print("Generate edge index and edge weight")
    edge_index, edge_weight, nodeList, nodeFeatures = getEdgeIndex()
    # nodeList = getEdgeNodeList()

    # dims = [32,64,128,256,512]
    # numberFetures = [8,16,32,64,128,256]
    for md in modelNames:
        for hiddenSize in dims:
            for numFeature in numberFetures:           
                # singleModel(edge_index, edge_weight, nodeList, nodeFeatures, md,numFeature, hiddenSize) 
                
                try:
                    singleModel(edge_index, edge_weight, nodeList, nodeFeatures, md,numFeature, hiddenSize) 
                except Exception as e:
                    print(e)
                    pass

                # try:
                #     thread1 = threading.Thread(target=singleModel, args=[edge_index, edge_weight, nodeList, nodeFeatures, md,numFeature, hiddenSize])
                #     # thread1.setDaemon(True)
                #     thread1.start()
                #     # thread1.join(timeout = 5)
                #     while(1):
                #         if(len(threading.enumerate()) < 64):
                #             break
                #         else:
                #             time.sleep(1)
                #         print("\r Thread left is {}:".format(len(threading.enumerate())),end="")
                # except Exception as e:
                #     print("{}: {} error, thread break!!!!!!!!!!!!!!!!!!!!!!!!".format(time.strftime("%D %H:%M:%S", time.localtime()), md))
                #     pass

        
        

if __name__ == "__main__":
    main()
