
import os
import sys
import pandas as pd
import numpy as np
import torch
import dgl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Project data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Import the original S-FFSD feature engineering function
sys.path.insert(0, os.path.join(BASE_DIR, "feature_engineering"))
from data_process import featmap_gen

def main():
    print("Starting S-FFSD preprocessing...")

    input_file = os.path.join(DATA_DIR, "S-FFSD", "S-FFSD.csv")
    output_file = os.path.join(DATA_DIR, "S-FFSDneofull.csv")

    # 1. Read original dataset
    print("Reading S-FFSD.csv...")
    data = pd.read_csv(input_file)
    data = data.reset_index(drop=True)

    # 2. Generate engineered features
    print("Generating engineered features...")
    data = featmap_gen(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(output_file, index=False)

    print("Saved S-FFSDneofull.csv")

    # 3. Build transaction graph
    print("Building S-FFSD graph...")
    data = data.reset_index(drop=True)

    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]

    for column in pair:
        src, tgt = [], []
        for _, group in tqdm(data.groupby(column), desc=column):
            group = group.sort_values(by="Time")
            indices = group.index
            n = len(indices)

            for i in range(n):
                for j in range(3):
                    if i + j < n:
                        src.append(indices[i])
                        tgt.append(indices[i + j])

        alls.extend(src)
        allt.extend(tgt)

    graph = dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(data))

    # 4. Encode categorical columns
    for column in pair:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column].astype(str))

    # 5. Add node labels and features
    labels = data["Labels"].to_numpy()
    features = data.drop("Labels", axis=1).to_numpy(dtype=np.float32)

    graph.ndata["label"] = torch.tensor(labels, dtype=torch.long)
    graph.ndata["feat"] = torch.tensor(features, dtype=torch.float32)

    # 6. Save graph
    graph_file = os.path.join(DATA_DIR, "graph-S-FFSD.bin")
    dgl.data.utils.save_graphs(graph_file, [graph])

    print("Saved graph-S-FFSD.bin")
    print("S-FFSD preprocessing completed successfully!")
    print("Number of transactions:", len(data))
    print("Number of graph edges:", graph.num_edges())

if __name__ == "__main__":
    main()
