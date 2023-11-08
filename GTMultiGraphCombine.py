# 将生成的multiGraph中的平行边合并
# weight值直接相加
# datetime 作为一个列表保存
# feature 相加

import pandas as pd
import numpy as np
import json
import networkx as nx
import argparse
# from ast import literal_eval

DiG = nx.DiGraph()
# To unify node ID and node label, add all node to Graph in advance
for i in range(2690):
    DiG.add_node(i, label = str(i))

G = nx.read_gml("结构熵代码panner\\moneyLunderingGraphGroundTruth.gml")


for u, v, key,data in G.edges(data=True, keys=True):
    u = int(u)
    v = int(v)
    if(u > 2689 or v > 2689):
        print(u,v,key,data["weight"])
    if(key==0):
        # add new edge
        DiG.add_edge(u,v,weight=data["weight"], datetime=str([data["datetime"]]),feature=data["feature"])
    else:
        # update edge value
        DiG[u][v]["weight"] += data["weight"]
        DiG[u][v]["datetime"] = str(eval(DiG[u][v]["datetime"]) + [data["datetime"]])
        DiG[u][v]["feature"] = str(list(np.array(eval(DiG[u][v]["feature"])) + np.array(eval(data["feature"]))))
        # print(DiG[u][v]["feature"])

nx.write_gml(DiG, "moneyLunderingDiGraphCombined.gml")
        
        