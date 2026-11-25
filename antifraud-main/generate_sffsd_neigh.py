
import os
import sys
import torch
import dgl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.insert(0, os.path.join(BASE_DIR, "feature_engineering"))
import data_process as dp
def main():
    print("Generating S-FFSD neighbour features...")

    graph_file = os.path.join(DATA_DIR, "graph-S-FFSD.bin")
    output_file = os.path.join(DATA_DIR, "S-FFSD_neigh_feat.csv")

    print("Loading S-FFSD graph...")
    graph = dgl.load_graphs(graph_file)[0][0]
    print(graph)

    print("Calculating degree features...")
    degree_feat = graph.in_degrees().unsqueeze(1).float()

    print("Calculating neighbour risk features...")
    risk_feat = dp.count_risk_neighs(graph).unsqueeze(1).float()

    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)

    print("Generating feature map...")
    dp.graph = graph
    dp.edge_feat = edge_feat

    features_neigh, feat_names = dp.feat_map()

    features_neigh = torch.cat(
        (edge_feat, features_neigh), dim=1
    ).numpy()

    feat_names = ["degree", "riskstat"] + feat_names
    features_neigh[np.isnan(features_neigh)] = 0.

    print("Scaling features...")
    features_neigh = pd.DataFrame(
        features_neigh, columns=feat_names
    )

    scaler = StandardScaler()
    features_neigh = pd.DataFrame(
        scaler.fit_transform(features_neigh),
        columns=features_neigh.columns
    )

    print("Saving neighbour features...")
    features_neigh.to_csv(output_file, index=False)

    print("Successfully generated:", output_file)
    print("Number of rows:", len(features_neigh))
    print("Number of columns:", len(features_neigh.columns))

if __name__ == "__main__":
    main()
