
import numpy as np
import pandas as pd
import torch
import dgl

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score
)

from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from methods.gtan.gtan_lpa import load_lpa_subtensor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEVICE = torch.device("cpu")
BATCH_SIZE = 128

CHECKPOINT = "models/gtan_trained_ckpt.pth"
FEATURE_FILE = "data/S-FFSD_feat_data.csv"
LABEL_FILE = "data/S-FFSD_label_data.csv"
GRAPH_FILE = "data/graph-S-FFSD.bin"

CAT_FEATURES = ["Target", "Location", "Type"]


# --------------------------------------------------
# 2. Load checkpoint
# --------------------------------------------------

print("\nLoading GTAN checkpoint...")

model = torch.load(CHECKPOINT, map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded:", type(model).__name__)


# --------------------------------------------------
# 3. Load original processed dataset
# --------------------------------------------------

print("\nLoading processed dataset...")

feat_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

labels_np = labels_df.iloc[:, 0].to_numpy()

print("Feature shape:", feat_df.shape)
print("Label shape:", labels_df.shape)


# --------------------------------------------------
# 4. Load original graph
# --------------------------------------------------

print("\nLoading graph...")

graphs, _ = dgl.load_graphs(GRAPH_FILE)
graph = graphs[0].to(DEVICE)

print("Graph nodes:", graph.num_nodes())
print("Graph edges:", graph.num_edges())


# --------------------------------------------------
# 5. Verify dataset consistency
# --------------------------------------------------

assert len(feat_df) == len(labels_np)
assert graph.num_nodes() == len(feat_df)
assert feat_df.shape[1] == model.in_feats

num_feat = torch.tensor(
    feat_df.values,
    dtype=torch.float32,
    device=DEVICE
)

graph_feat = graph.ndata["feat"]

assert torch.allclose(num_feat, graph_feat)

print("Feature and graph consistency: PASSED")


# --------------------------------------------------
# 6. Prepare categorical features
# --------------------------------------------------

print("\nPreparing categorical features...")

cat_feat = {
    col: torch.tensor(
        feat_df[col].values,
        dtype=torch.long,
        device=DEVICE
    )
    for col in CAT_FEATURES
}

for col in CAT_FEATURES:

    stored = model.n2v_mlp.cat_features[col].cpu()

    mapping = dict(
        zip(feat_df[col].values, stored.tolist())
    )

    mapped = feat_df[col].map(mapping).to_numpy()

    assert np.array_equal(mapped, stored.numpy())

    print(col, "mapping verified")


# --------------------------------------------------
# 7. Prepare labels
# --------------------------------------------------

labels = torch.tensor(
    labels_np,
    dtype=torch.long,
    device=DEVICE
)


# --------------------------------------------------
# 8. Recreate original test split
# --------------------------------------------------

print("\nRecreating original test split...")

indices = np.arange(len(labels_np))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=2,
    shuffle=True,
    stratify=labels_np
)

print("Training transactions:", len(train_idx))
print("Testing transactions:", len(test_idx))


# --------------------------------------------------
# 9. Create graph sampler
# --------------------------------------------------

sampler = MultiLayerFullNeighborSampler(model.n_layers)

test_ind = torch.tensor(
    test_idx,
    dtype=torch.long,
    device=DEVICE
)

test_loader = NodeDataLoader(
    graph,
    test_ind,
    sampler,
    device=DEVICE,
    use_ddp=False,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=0
)


# --------------------------------------------------
# 10. Run GTAN inference
# --------------------------------------------------

print("\nStarting GTAN inference...")

test_predictions = torch.zeros(
    (len(feat_df), 2),
    dtype=torch.float32,
    device=DEVICE
)

model.eval()

with torch.no_grad():

    for step, (input_nodes, seeds, blocks) in enumerate(test_loader):

        batch_inputs, batch_work_inputs, batch_labels, lpa_labels = (
            load_lpa_subtensor(
                num_feat,
                cat_feat,
                labels,
                seeds,
                input_nodes,
                DEVICE
            )
        )

        blocks = [block.to(DEVICE) for block in blocks]

        logits = model(
            blocks,
            batch_inputs,
            lpa_labels,
            batch_work_inputs
        )

        test_predictions[seeds] = logits

        if step % 10 == 0:
            print("Completed test batch:", step)


# --------------------------------------------------
# 11. Calculate evaluation metrics
# --------------------------------------------------

print("\nCalculating evaluation metrics...")

test_scores = torch.softmax(
    test_predictions[test_idx],
    dim=1
)[:, 1].cpu().numpy()

test_predicted_labels = torch.argmax(
    test_predictions[test_idx],
    dim=1
).cpu().numpy()

actual_labels = labels_np[test_idx]

# Exclude label 2, as in original GTAN evaluation
mask = actual_labels != 2

actual_labels = actual_labels[mask]
test_scores = test_scores[mask]
test_predicted_labels = test_predicted_labels[mask]

auc = roc_auc_score(actual_labels, test_scores)

f1 = f1_score(
    actual_labels,
    test_predicted_labels,
    average="macro"
)

ap = average_precision_score(
    actual_labels,
    test_scores
)


# --------------------------------------------------
# 12. Display results
# --------------------------------------------------

print("\n===================================")
print("       GTAN INFERENCE RESULTS")
print("===================================")

print("Evaluated transactions:", len(actual_labels))
print("Fraud transactions:", np.sum(actual_labels == 1))
print("Legitimate transactions:", np.sum(actual_labels == 0))

print("-----------------------------------")
print("Test AUC:", round(auc, 6))
print("Test F1 :", round(f1, 6))
print("Test AP :", round(ap, 6))
print("===================================")

print("\nRecorded original GTAN results:")
print("Test AUC: 0.8089388696")
print("Test F1 : 0.7447966658")
print("Test AP : 0.6478554548")

print("\nInference testing completed.")
