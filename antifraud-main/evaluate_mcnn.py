
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

DEVICE = torch.device("cpu")
CHECKPOINT = "models/mcnn_ckpt.pth"

print("Loading MCNN test data...")

test_features = np.load("data/tef_2d.npy", allow_pickle=True)
test_labels = np.load("data/tel_2d.npy", allow_pickle=True)

print("Test features:", test_features.shape)
print("Test labels:", test_labels.shape)

# Match the original MCNN preprocessing.
test_features = torch.tensor(
    test_features, dtype=torch.float32, device=DEVICE
)
test_features = test_features.transpose(1, 2)

test_labels = torch.tensor(
    test_labels, dtype=torch.long, device=DEVICE
)

print("\nLoading trained MCNN checkpoint...")
model = torch.load(CHECKPOINT, map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

print("MCNN model loaded successfully!")
print("\nRunning MCNN inference...")

all_logits = []

with torch.no_grad():
    batch_size = 512

    for start in range(0, len(test_labels), batch_size):
        end = min(start + batch_size, len(test_labels))
        logits = model(test_features[start:end])
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)

    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = torch.argmax(logits, dim=1)

true_labels = test_labels.cpu().numpy()
scores = probabilities.cpu().numpy()
predicted_labels = predictions.cpu().numpy()

auc = roc_auc_score(true_labels, scores)
f1 = f1_score(true_labels, predicted_labels, average="macro")
ap = average_precision_score(true_labels, scores)

print("\n========== MCNN INFERENCE RESULTS ==========")
print("Evaluated transactions:", len(true_labels))
print("Fraud transactions:", int((true_labels == 1).sum()))
print("Legitimate transactions:", int((true_labels == 0).sum()))
print("Test AUC: {:.6f}".format(auc))
print("Test F1 : {:.6f}".format(f1))
print("Test AP : {:.6f}".format(ap))

with open("results/mcnn_inference_results.txt", "w") as f:
    f.write("MCNN Inference Results\n")
    f.write("AUC: {:.6f}\n".format(auc))
    f.write("F1: {:.6f}\n".format(f1))
    f.write("AP: {:.6f}\n".format(ap))
    f.write("Evaluated transactions: {}\n".format(len(true_labels)))

print("\nResults saved to results/mcnn_inference_results.txt")
print("MCNN inference completed successfully!")
