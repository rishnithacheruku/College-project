
import torch
import numpy as np
import pandas as pd
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from methods.rgtan.rgtan_main import loda_rgtan_data
from methods.rgtan.rgtan_lpa import load_lpa_subtensor

DEVICE = "cpu"
CHECKPOINT = "models/rgtan_ckpt.pth"
BATCH_SIZE = 64

print("Loading RGTAN test data...")

feat_data, labels, train_idx, test_idx, graph, cat_features, neigh_features = (
    loda_rgtan_data("S-FFSD", 0.6)
)

graph = graph.to(DEVICE)

num_feat = torch.from_numpy(feat_data.values).float().to(DEVICE)

cat_feat = {
    col: torch.from_numpy(feat_data[col].values).long().to(DEVICE)
    for col in cat_features
}

if isinstance(neigh_features, pd.DataFrame):
    nei_feat = {
        col: torch.from_numpy(neigh_features[col].values).float().to(DEVICE)
        for col in neigh_features.columns
    }
else:
    nei_feat = {}

neigh_padding_dict = {}
labels_tensor = torch.from_numpy(labels.values).long().to(DEVICE)

print("Loading trained RGTAN checkpoint...")

model = torch.load(CHECKPOINT, map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

print("RGTAN model loaded successfully!")
print("Running RGTAN inference...")

test_ind = torch.from_numpy(np.asarray(test_idx)).long().to(DEVICE)

sampler = MultiLayerFullNeighborSampler(model.n_layers)

test_dataloader = NodeDataLoader(
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

predictions = torch.zeros((len(feat_data), 2), device=DEVICE)

with torch.no_grad():
    for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):

        batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = (
            load_lpa_subtensor(
                num_feat,
                cat_feat,
                nei_feat,
                neigh_padding_dict,
                labels_tensor,
                seeds,
                input_nodes,
                DEVICE,
                blocks
            )
        )

        blocks = [block.to(DEVICE) for block in blocks]

        logits = model(
            blocks,
            batch_inputs,
            lpa_labels,
            batch_work_inputs,
            batch_neighstat_inputs
        )

        predictions[seeds] = logits

        if (step + 1) % 20 == 0:
            print("Processed batches:", step + 1)

print("\nCalculating evaluation metrics...")

test_scores = torch.softmax(predictions, dim=1)[test_ind, 1].cpu().numpy()
test_predictions = torch.argmax(predictions, dim=1)[test_ind].cpu().numpy()
test_labels = labels_tensor[test_ind].cpu().numpy()

# Exclude class 2, as done in the original RGTAN evaluation.
mask = test_labels != 2

test_scores = test_scores[mask]
test_predictions = test_predictions[mask]
test_labels = test_labels[mask]

auc = roc_auc_score(test_labels, test_scores)
f1 = f1_score(test_labels, test_predictions, average="macro")
ap = average_precision_score(test_labels, test_scores)

print("\n========== RGTAN INFERENCE RESULTS ==========")
print("Evaluated transactions:", len(test_labels))
print("Fraud transactions:", int((test_labels == 1).sum()))
print("Legitimate transactions:", int((test_labels == 0).sum()))
print("Test AUC:", round(auc, 6))
print("Test F1 :", round(f1, 6))
print("Test AP :", round(ap, 6))

with open("results/rgtan_inference_results.txt", "w") as f:
    f.write("RGTAN INFERENCE RESULTS\n")
    f.write(f"Evaluated transactions: {len(test_labels)}\n")
    f.write(f"Fraud transactions: {int((test_labels == 1).sum())}\n")
    f.write(f"Legitimate transactions: {int((test_labels == 0).sum())}\n")
    f.write(f"Test AUC: {auc:.6f}\n")
    f.write(f"Test F1: {f1:.6f}\n")
    f.write(f"Test AP: {ap:.6f}\n")

print("\nResults saved to results/rgtan_inference_results.txt")
print("RGTAN inference completed successfully!")
