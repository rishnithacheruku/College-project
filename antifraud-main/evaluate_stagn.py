
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from methods.stagn.stagn_main import load_stagn_data

DEVICE = torch.device("cpu")
CHECKPOINT = "models/stagn_ckpt.pth"

print("Loading STAGN test data...")
features, labels, graph = load_stagn_data({
    "test_size": 0.6,
    "method": "STAGN"
})

print("Features:", features.shape)
print("Labels:", labels.shape)

# Recreate the original STAGN split procedure.
train_idx, test_idx = train_test_split(
    np.arange(features.shape[0]),
    test_size=0.6,
    stratify=labels
)

print("Test transactions:", len(test_idx))

print("\nLoading trained STAGN checkpoint...")
model = torch.load(CHECKPOINT, map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

# Use the graph reconstructed from the original dataset.
graph = graph.to(DEVICE)

# Initialize graph node features as expected by TransactionGCN.
# The graph convolution uses the edge feature dimension.
with torch.no_grad():
    graph.ndata["feat"] = torch.zeros(
        graph.num_nodes(),
        graph.edata["feat"].shape[1],
        dtype=torch.float32,
        device=DEVICE
    )

    # Match the feature orientation used during original training.
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    x = x.transpose(1, 2)

    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    print("Running STAGN inference...")
    logits = model(x, graph)

    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = torch.argmax(logits, dim=1)

    test_labels = y[test_idx].cpu().numpy()
    test_scores = probabilities[test_idx].cpu().numpy()
    test_predictions = predictions[test_idx].cpu().numpy()

auc = roc_auc_score(test_labels, test_scores)
f1 = f1_score(test_labels, test_predictions, average="macro")
ap = average_precision_score(test_labels, test_scores)

print("\n========== STAGN INFERENCE RESULTS ==========")
print("Evaluated transactions:", len(test_idx))
print("Fraud transactions:", int((test_labels == 1).sum()))
print("Legitimate transactions:", int((test_labels == 0).sum()))
print("Test AUC: {:.6f}".format(auc))
print("Test F1 : {:.6f}".format(f1))
print("Test AP : {:.6f}".format(ap))

with open("results/stagn_inference_results.txt", "w") as f:
    f.write("STAGN Inference Results\n")
    f.write("AUC: {:.6f}\n".format(auc))
    f.write("F1: {:.6f}\n".format(f1))
    f.write("AP: {:.6f}\n".format(ap))
    f.write("Evaluated transactions: {}\n".format(len(test_idx)))

print("\nResults saved to results/stagn_inference_results.txt")
print("STAGN inference completed!")
