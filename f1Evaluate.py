import networkx as nx
import os
import fnmatch
import pandas as pd

DG = nx.read_gml("combined_20220729_lower300000.gml")
mapping = dict(zip(DG.nodes(), list(map(int, DG.nodes()))))
DG = nx.relabel_nodes(DG, mapping)
for u, v in DG.edges:
    DG[u][v]["weight"] = float(DG[u][v]["weight"])

# # grountruth
# with open("gt_20210908.txt") as f:
#     gt = [line.strip() for line in f.readlines()]
    
# grountruth
with open("gt_20220729_lower300000.txt") as f:
    gt = eval(f.readline())

gt = list(map(int, gt))

modelNameDict = {
    "SEComm": "SEComm",
    "deepwalk": "DeepWalk",
    "hope": "HOPE",
    "line": "Line",
    "netmf": "NetMF",
    "netsmf": "NetSMF",
    "node2vec": "node2vec",
    "prone": "ProNE",
    "spectral": "SocioDim",
    "dgi": "DGI",
    "vgae": "VGAE",
    "FeatureOnly": "FeatureOnly",
    "dgi" : "DGI",
    "mvgrl":"MVGRL", 
    "grace":"GRACE", 
    "unsup_graphsage":"GraphSAGE"
}

# community
# 把算法得到的社区中的节点编号存到seq列表中
def evaluate(resFileName):
    communityRes = [[] for i in range(31)]
    with open(resFileName) as f:
        for line in f.readlines():
            node, id = line.split(" ")
            id = int(id)
            node = int(node)
            communityRes[id].append(node)
    # seq = []

    pr = nx.pagerank(DG)

    # generate panda table
    modelnameList = []
    classifierList = []
    featureList = []
    dimList = []
    recall_piList = []
    precision_piList = []
    recall_nodeNumList = []
    precision_nodeNumList = []
    resSumList = []
    f1_scoreList = []
    communityCountList = []
    splitCharList = []

    for i, seq in enumerate(communityRes):
        # compute pi
        pi_gt = 0
        pi_seq = 0
        pi_right = 0
        for node in gt:
            if node in pr.keys():
                pi_gt += pr[node]
        for node in seq:
            if node in pr.keys():
                pi_seq += pr[node]
        for node in set(gt) & set(seq):
            if node in pr.keys():
                pi_right += pr[node]

        if pi_gt != 0:
            recall_pi = pi_right / pi_gt
        else:
            recall_pi = 0
        if pi_seq != 0:
            precision_pi = pi_right / pi_seq
        else:
            precision_pi = 0

        num_seq = len(seq)
        num_gt = len(gt)
        num_right = len(set(gt) & set(seq))

        # calculate F-1 Score:
        TruePositive = len(set(gt) & set(seq))
        FalsePositive = len(set(seq) - set(gt))
        FalseNegative = len(set(gt) - set(seq))
        f1_score = 2 * TruePositive / (2 * TruePositive + FalsePositive + FalseNegative)
        # print(num_seq, num_gt, num_right)

        if num_gt != 0:
            recall_nodeNum = num_right / num_gt
        else:
            recall_nodeNum = 0

        if num_seq != 0:
            precision_nodeNum = num_right / num_seq
        else:
            precision_nodeNum = 0

        resSum = recall_pi + precision_pi + recall_nodeNum + precision_nodeNum + f1_score

        # "combined_20210908.gml_prone_kmeans_feature_8_dim_32.res"
        fileInfo = resFileName.split(".")[1]
        modelname = fileInfo.split("_")[1]
        classifier = fileInfo.split("_")[2]
        feature = fileInfo.split("_")[4]
        dim = fileInfo.split("_")[6]
        # print(recall_pi, precision_pi)
        # print(
        #     "{} \t {} \t {} \t {} \t {:.3} \t {:.3} \t {:.3} \t {:.3} \t {:.3} \t {:.3}".format(
        #         modelname,
        #         classifier,
        #         feature,
        #         dim,
        #         recall_pi,
        #         precision_pi,
        #         recall_nodeNum,
        #         precision_nodeNum,
        #         resSum,
        #         f1_score,
        #     )
        # )
        modelnameList.append(modelNameDict[modelname])
        classifierList.append(classifier)
        featureList.append(feature)
        dimList.append(dim)
        recall_piList.append(recall_pi)
        precision_piList.append(precision_pi)
        recall_nodeNumList.append(recall_nodeNum)
        precision_nodeNumList.append(precision_nodeNum)
        resSumList.append(resSum)
        f1_scoreList.append(f1_score)
        communityCountList.append(num_seq)
        splitCharList.append('&')

    resPdTable = pd.DataFrame(
        {
            "modelname": modelnameList,
            "sp1":splitCharList,
            "classifier": classifierList,
            "sp2":splitCharList,
            "feature": featureList,
            "sp3":splitCharList,
            "dim": dimList,
            "sp4":splitCharList,
            "recall_pi": recall_piList,
            "sp5":splitCharList,
            "precision_pi": precision_piList,
            "sp6":splitCharList,
            "recall_nodeNum": recall_nodeNumList,
            "sp7":splitCharList,
            "precision_nodeNum": precision_nodeNumList,
            "sp8":splitCharList,
            "resSum": resSumList,
            "sp9":splitCharList,
            "f1_score": f1_scoreList,
            "sp10":splitCharList,
            "communityCount": communityCountList,
        }
    )
    return resPdTable


def main():
    # print(
    #     "{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format(
    #         "modelname",
    #         "classifier",
    #         "feature",
    #         "dim",
    #         "recall_pi",
    #         "precision_pi",
    #         "recall_nodeNum",
    #         "precision_nodeNum",
    #         "resSum",
    #         "f1-score",
    #     )
    # )
    # print(fnmatch.filter(os.listdir(), "*.res"))
    pdList = []
    for x in fnmatch.filter(os.listdir(), "*.res"):
        pdList.append(evaluate(x))
    pf = pd.concat(pdList)
    # print(pf)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", 200)
    pd.set_option("display.width",200)

    group = (
        pf.sort_values("f1_score", ascending=False)
        .groupby("modelname", as_index=True)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)                                 
    )
    # group = pf.groupby("modelname", as_index=False).first()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(
    #     group[
    #         [
    #             "modelname",
    #             "classifier",
    #             "feature",
    #             "dim",
    #             "recall_pi",
    #             "precision_pi",
    #             "recall_nodeNum",
    #             "precision_nodeNum",
    #             "resSum",
    #             "f1_score",
    #             "communityCount",
    #         ]
    #     ]
    # )
    print(
        group[
            [
                "modelname",
                "sp1",
                "feature",
                "sp1",
                "dim",
                "sp1",
                "recall_pi",
                "sp1",
                "precision_pi",
                "sp1",
                "recall_nodeNum",
                "sp1",
                "precision_nodeNum",
                "sp1",
                "f1_score",
                "sp1",
                "communityCount",
            ]
        ]
    )

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    group = (
        pf.sort_values("recall_pi", ascending=False)
        .groupby("modelname", as_index=False)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)
    )
    # print(group[["modelname", "classifier", "feature", "dim", "recall_pi", "communityCount"]])
    print(group[["modelname", "sp1", "feature", "sp1","dim", "sp1","recall_pi"]])

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    group = (
        pf.sort_values("precision_pi", ascending=False)
        .groupby("modelname", as_index=False)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)
    )
    # print(group[["modelname", "classifier", "feature", "dim", "precision_pi", "communityCount"]])
    print(group[["modelname", "sp1", "feature", "sp1","dim", "sp1","precision_pi"]])

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    group = (
        pf.sort_values("recall_nodeNum", ascending=False)
        .groupby("modelname", as_index=False)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)
    )
    # print(group[["modelname", "classifier", "feature", "dim", "recall_nodeNum", "communityCount"]])
    print(group[["modelname", "sp1", "feature", "sp1","dim", "sp1","recall_nodeNum"]])

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    group = (
        pf.sort_values("precision_nodeNum", ascending=False)
        .groupby("modelname", as_index=False)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)
    )
    # print(group[["modelname", "classifier", "feature", "dim", "precision_nodeNum", "communityCount"]])
    print(group[["modelname", "sp1", "feature", "sp1","dim", "sp1","precision_nodeNum"]])

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    group = (
        pf.sort_values("f1_score", ascending=False)
        .groupby("modelname", as_index=False)
        .head(1)
        .sort_values("modelname", ascending=True)
        .round(3)
    )
    # print(group[["modelname", "classifier", "feature", "dim", "f1_score", "communityCount"]])
    print(group[["modelname", "sp1", "feature", "sp1","dim", "sp1","f1_score"]])

    group = (
        pf.sort_values("f1_score", ascending=False)
        .groupby(["modelname", "feature", "dim"], as_index=True)
        .head(21)
        .round(3)
    )
    # group = pf.groupby("modelname", as_index=False).first()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(group[["modelname", "classifier", "feature", "dim","f1_score", "communityCount"]])
    # print(group[["modelname", "classifier", "feature", "dim","recall_pi","sp1","precision_pi","sp1","recall_nodeNum","sp1","precision_nodeNum", "f1_score", "communityCount"]])


if __name__ == "__main__":
    main()
