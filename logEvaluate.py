import networkx as nx
import os
import fnmatch
import pandas as pd
import numpy as np

def main():
    logFile = "local_structural_entropy_onlyentropyBetaRatioTest.csv"
    logpd = pd.read_csv(logFile, names = ['embeddingFileName', 'beta', 'maxF1', 'maxF1_recall', 'maxF1_precision', 'maxF1_communitySize', 'f1_score', 'recall_nodeNum', 'precision_nodeNum', 'time'])
    logpd = logpd.drop(labels='embeddingFileName',axis=1)
    print(logpd)
    group = (
    logpd.sort_values("maxF1", ascending=False)
    .groupby("beta", as_index=True)
    .head(1)
    .sort_values("beta", ascending=True)
    .round(3)                                 
    )
    print(group)
    group.to_csv("onlyentropyBetaRatioTestResult.csv",index=False)

if __name__ == "__main__":
    main()

