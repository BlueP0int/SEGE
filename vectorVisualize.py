import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import numpy as np
import os
import fnmatch
import argparse
from random import sample
import networkx as nx

# embeddingFileName = 'combined_20210928_connected.gml_spectral_feature_128_dim_64.emb'

# # grountruth
# with open("/home/pc/cogdl/examples/gt_20210908.txt") as f:
#     gt = [line.strip() for line in f.readlines()]


gmlFileName = 'combined_20220729_lower300000.gml'

def embVisualize(embeddingFileName, gt, entropyRes, entropyEmbRes):
    embList = []
    gtColorList = ['c' for i in range(54227)]
    entropyColorList = ['c' for i in range(54227)]
    entropyEmbColorList = ['c' for i in range(54227)]

    for i in gt:
        gtColorList[int(i)] = 'r'
    for i in entropyRes:
        entropyColorList[int(i)] = 'b'
    for i in entropyEmbRes:
        entropyEmbColorList[int(i)] = 'g'

    with open(embeddingFileName, "r") as f:
        for line in f.readlines():
            # if(random.randint(0,101) <= 100):
            nodeNum,vec = line.split(':')
            # # nodeNum = int(line.split(':')[0].strip())
            # if(nodeNum in marklist):
            #     # colorList.append('r')  # for gt
            #     colorList.append('b')    # for entropy
            #     # colorList.append('g')    # for entropy + embedding
            # else:
            #     colorList.append('c')
            vec = vec.strip().split()
            embList.append(list(map(float, vec)))
    embeddings = np.array(embList)

    if(embeddings.shape[0] != len(gtColorList)):
        return
    else:
        print("size ok!")


    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(embeddings)

    # reduced = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(embeddings)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    reduced = tsne.fit_transform(embeddings)

    t = reduced.transpose()
    plt.scatter(t[0], t[1],s=1,c=gtColorList,linewidths=0)
    # plt.title(embeddingFileName)
    plt.savefig(embeddingFileName + '_GroundTruth.jpg', format='jpg', dpi=800)
    print('saved '+ embeddingFileName + '_GroundTruth.jpg')

    plt.clf()
    plt.scatter(t[0], t[1],s=1,c=entropyColorList,linewidths=0)
    plt.savefig(embeddingFileName + '_EntropyRes.jpg', format='jpg', dpi=800)
    print('saved '+ embeddingFileName + '_EntropyRes.jpg')

    plt.clf()
    plt.scatter(t[0], t[1],s=1,c=entropyEmbColorList,linewidths=0)
    plt.savefig(embeddingFileName + '_EntropyEmbedding.jpg', format='jpg', dpi=800)
    print('saved '+ embeddingFileName + '_EntropyEmbedding.jpg')


def embVisualize2(embeddingFileName, gt, entropyRes, embFiles):
    embeddingFileName = "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_32.emb"
    embList = []
    with open(embeddingFileName, "r") as f:
        for line in f.readlines():
            nodeNum,vec = line.split(':')
            vec = vec.strip().split()
            embList.append(list(map(float, vec)))
    embeddings = np.array(embList)

    # if(embeddings.shape[0] != len(gtColorList)):
    #     return
    # else:
    #     print("size ok!")

    # reduced = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(embeddings)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    reduced = tsne.fit_transform(embeddings)

    baseColorList = ['lightgray' for i in range(54227)]
    baseNodeRList = [1 for i in range(54227)]
    
    gtColorList = ['lightgray' for i in range(54227)]
    entropyColorList = ['lightgray' for i in range(54227)]
    gtNodeRList = [0 for i in range(54227)]
    entropyNodeRList = [0 for i in range(54227)]
    

    for i in gt:
        gtColorList[int(i)] = 'b'
        gtNodeRList[int(i)] = 5
    # 预测正确的点（TRUE positive）
    for i in list(set(gt).intersection(set(entropyRes))):
        entropyColorList[int(i)] = 'b'
        entropyNodeRList[int(i)] = 5
    # false positive
    for i in list(set(entropyRes).difference(set(gt))):
        entropyColorList[int(i)] = 'r'
        entropyNodeRList[int(i)] = 4
    # false negative
    for i in list(set(gt).difference(set(entropyRes))):
        entropyColorList[int(i)] = 'r'
        entropyNodeRList[int(i)] = 4

    t = reduced.transpose()
    plt.axis('off')
    plt.scatter(t[0], t[1], s=baseNodeRList, c=baseColorList,linewidths=0)
    plt.scatter(t[0], t[1], s=gtNodeRList, c=gtColorList,linewidths=0)

    # plt.title(embeddingFileName)
    plt.savefig(embeddingFileName + '_GroundTruth.png', format='png', dpi=800)
    print('saved '+ embeddingFileName + '_GroundTruth.png')

    plt.clf()
    plt.axis('off')
    plt.scatter(t[0], t[1], s=baseNodeRList, c=baseColorList,linewidths=0)
    plt.scatter(t[0], t[1], s=entropyNodeRList, c=entropyColorList,linewidths=0)
    plt.savefig(embeddingFileName + '_EntropyRes.png', format='png', dpi=800)
    print('saved '+ embeddingFileName + '_EntropyRes.png')


    entropyEmbColorList = ['lightgray' for i in range(54227)]
    entropyEmbNodeRList = [0 for i in range(54227)]
    # entropyEmbResFileName = fname[:-4] + ".res"
    entropyEmbResFileName = 'combined_20220729_lower300000.gml_node2vec_feature_-1_dim_32.emb_SEntropy_Embedding.res'

    # graphEmbRes = []
    # with open(entropyEmbResFileName) as f:
    #     for line in f.readlines():
    #         node, id = line.split(" ")
    #         id = int(id)
    #         # node = int(node)
    #         if(id == communityID[i]):
    #             graphEmbRes.append(node)
    with open(entropyEmbResFileName) as f:
        entropyEmbRes = eval(f.readline())
    for i in list(set(gt).intersection(set(entropyEmbRes))):
        entropyEmbColorList[int(i)] = 'b' 
        entropyEmbNodeRList[int(i)] = 5
    # false positive
    for i in list(set(entropyEmbRes).difference(set(gt))):
        entropyEmbColorList[int(i)] = 'r'
        entropyEmbNodeRList[int(i)] = 4
    # false negative
    for i in list(set(gt).difference(set(entropyEmbRes))):
        entropyEmbColorList[int(i)] = 'r'
        entropyEmbNodeRList[int(i)] = 4

    plt.clf()
    plt.axis('off')
    plt.scatter(t[0], t[1], s=baseNodeRList, c=baseColorList,linewidths=0)
    plt.scatter(t[0], t[1],s=entropyEmbNodeRList,c=entropyEmbColorList,linewidths=0)
    plt.savefig(entropyEmbResFileName + '_GraphEmbeddingKMeans.png', format='png', dpi=800)
    print('saved '+ entropyEmbResFileName + '_GraphEmbeddingKMeans.png')

    # community个数设置为30
    communityID = [14, 1, 19, 5, 0, 27, 12, 14, 9]
    for i, fname in enumerate(embFiles):
        entropyEmbColorList = ['lightgray' for i in range(54227)]
        entropyEmbNodeRList = [0 for i in range(54227)]
        # entropyEmbResFileName = fname[:-4] + ".res"
        entropyEmbResFileName = fname.replace('.emb', '.res').replace('feature', 'kmeans_feature')

        graphEmbRes = []
        with open(entropyEmbResFileName) as f:
            for line in f.readlines():
                node, id = line.split(" ")
                id = int(id)
                # node = int(node)
                if(id == communityID[i]):
                    graphEmbRes.append(node)
        # with open(entropyEmbResFileName) as f:
        #     entropyEmbRes = eval(f.readline())
        for i in list(set(gt).intersection(set(graphEmbRes))):
            entropyEmbColorList[int(i)] = 'b' 
            entropyEmbNodeRList[int(i)] = 5
        # false positive
        for i in list(set(graphEmbRes).difference(set(gt))):
            entropyEmbColorList[int(i)] = 'r'
            entropyEmbNodeRList[int(i)] = 4
        # false negative
        for i in list(set(gt).difference(set(graphEmbRes))):
            entropyEmbColorList[int(i)] = 'r'
            entropyEmbNodeRList[int(i)] = 4

        plt.clf()
        plt.axis('off')
        plt.scatter(t[0], t[1], s=baseNodeRList, c=baseColorList,linewidths=0)
        plt.scatter(t[0], t[1],s=entropyEmbNodeRList,c=entropyEmbColorList,linewidths=0)
        plt.savefig(fname + '_GraphEmbeddingKMeans.png', format='png', dpi=800)
        print('saved '+ fname + '_GraphEmbeddingKMeans.png')


def networkxVisualize(gt, entropyRes, entropyEmbRes):
    G = nx.read_gml(gmlFileName)
    gtSample = sample(gt, int(len(gt)*0.5))
    nodeSample = sample(list(G.nodes), int(len(list(G.nodes))*0.5))
    subGnodes = list(set(gtSample + nodeSample))
    subG = G.subgraph(subGnodes)
    colorList = []
    for node in list(subG.nodes):
        if node in gtSample:
            colorList.append('r')
        else:
            colorList.append('c')
    nx.draw(subG,
        pos = nx.random_layout(subG), # pos 指的是布局,主要有spring_layout,random_layout,circle_layout,shell_layout
        node_color = colorList,   # node_color指节点颜色,有rbykw,同理edge_color 
        edge_color = 'r',
        with_labels = False,  # with_labels指节点是否显示名字
        width =0,  # font_size表示字体大小,font_color表示字的颜色
        arrowsize = 0,
        node_size =1)  # font_size表示字体大小,font_color表示字的颜色
    plt.savefig("network.png")
    nx.write_gexf(subG, 'network.gexf')  # gexf格式文件可以导入gephi中进行分析
    plt.show()

def main():
    embFiles = ["combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_node2vec_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_line_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_dgi_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_128.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_32.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_prone_feature_-1_dim_64.emb"]
    
    embFiles = ["combined_20220729_lower300000.gml_dgi_feature_-1_dim_128.emb","combined_20220729_lower300000.gml_deepwalk_feature_-1_dim_64.emb", "combined_20220729_lower300000.gml_hope_feature_-1_dim_64.emb","combined_20220729_lower300000.gml_line_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_mvgrl_feature_-1_dim_256.emb", "combined_20220729_lower300000.gml_netmf_feature_-1_dim_512.emb", "combined_20220729_lower300000.gml_netsmf_feature_-1_dim_64.emb",  "combined_20220729_lower300000.gml_prone_feature_-1_dim_512.emb","combined_20220729_lower300000.gml_node2vec_feature_-1_dim_32.emb"]
    parser = argparse.ArgumentParser()
    # 44 embedding files in total, so the embeddingPtr should in [0,43]
    parser.add_argument("--embeddingPtr", type=int, default=0)
    args = parser.parse_args()

    embeddingFileName = embFiles[args.embeddingPtr]
    entropyEmbResFileName = embeddingFileName + "_SEntropy_Embedding.res"
    # grountruth
    with open("gt_20220729_lower300000.txt") as f:
        gt = eval(f.readline())

    # entropy only result
    with open("combined_20220729_lower300000.gml_SEntropy.res") as f:
        entropyRes = eval(f.readline())

    # entropy and embedding result
    with open(entropyEmbResFileName) as f:
        entropyEmbRes = eval(f.readline())

    embVisualize2(embeddingFileName, gt, entropyRes, embFiles)

    # networkxVisualize(gt, entropyRes, entropyEmbRes)


if __name__ == "__main__":
    main()